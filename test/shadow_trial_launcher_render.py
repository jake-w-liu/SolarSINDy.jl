"""Exercise extracted maintained render definitions with independent literal plists.

No installer top level, service loading function or HOME override is executed.
The private lint stub only records calls; plistlib independently parses XML.
"""
from pathlib import Path
import argparse,hashlib,json,os,plistlib,re,subprocess

def shell_quote(value):return "'"+str(value).replace("'","'\"'\"'")+"'"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',required=True);parser.add_argument('--bash',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();repo=Path(args.repo).resolve();output=Path(args.output).resolve();output.mkdir(exist_ok=False)
    source=repo/'deploy/install_launchd.sh';body=source.read_text(encoding='utf-8')
    oracle=repo/'test/shadow_trial_launcher_oracle.json';fixed=json.loads(oracle.read_bytes())
    names=['xml_escape','xml_sed_escape','service_env_keys','service_suffix','write_extra_env_block','render']
    functions=[]
    for name in names:
        matches=list(re.finditer(r'^'+name+r'\(\) \{\n.*?^}\n',body,re.M|re.S))
        assert len(matches)==1,(name,len(matches))
        functions.append(matches[0].group())
    known=re.search(r'^KNOWN_CONFIG_KEYS="[^"]*"',body,re.M).group()
    diagnostic=body[body.index('UNAPPLIED_KEYS=""'):body.index('\nif [ -n "${SOLARSINDY_LOAD+x}" ]; then')]
    baseline_env=dict(os.environ)
    for key in re.findall('[A-Z][A-Z_0-9]*',known)+fixed['trial_keys']:baseline_env.pop(key,None)
    record=dict(complete=False,checks=[],source_sha256=sha(source),oracle_sha256=sha(oracle),installer_executed=False,launchctl_executed=False,home_modified=False)
    def check(value,name):record['checks'].append(dict(name=name,passed=bool(value)))
    try:
        paths=fixed['paths'];keys=fixed['trial_keys']
        cases=[('absent',{}),('all',fixed['trial_values']),('config-only',{keys[0]:fixed['trial_values'][keys[0]]}),
            ('config-pin-only',{keys[1]:fixed['trial_values'][keys[1]]}),('seal-pin-only',{keys[2]:fixed['trial_values'][keys[2]]}),('empty-config',{keys[0]:''})]
        for name,trial_values in cases:
            directory=output/name;directory.mkdir();work=directory/'render-work';work.mkdir();installed=directory/'private-plists';installed.mkdir()
            assignments=dict(SELF_DIR=(repo/'deploy').as_posix(),WORK_DIR=work.as_posix(),LA_DIR=installed.as_posix(),ORG='trial-render',
                JULIA_BIN=paths['julia'],CLONE_DIR=paths['clone'],APP_DIR=paths['app'],MONITOR_DIR=paths['monitor'],
                RECEIPT_DIR=paths['receipts'],RECEIPT_LOG_DIR=paths['logs'],SWM_HOST_VALUE=paths['host'],SWM_PORT_VALUE=paths['port'],
                WATCHDOG_DASH_URL_VALUE=paths['health'],WATCHDOG_STALE_SEC_VALUE=paths['stale'])
            shell='set -euo pipefail\n'+'\n'.join(functions)+'\n'
            shell+='\n'.join(key+'='+shell_quote(value) for key,value in assignments.items())+'\n'
            shell+='plutil() { [ "$1" = -lint ] && [ -f "$2" ]; printf "lint\\n" >> "$WORK_DIR/lint.calls"; }\n'
            shell+='launchctl() { printf "unexpected launchctl\\n" > "$WORK_DIR/launchctl.called"; return 91; }\n'
            for target,key in [('JULIA_RENDER','JULIA_BIN'),('CLONE_RENDER','CLONE_DIR'),('APP_RENDER','APP_DIR'),('MONITOR_RENDER','MONITOR_DIR'),('RECEIPT_RENDER','RECEIPT_DIR'),('RECEIPT_LOG_RENDER','RECEIPT_LOG_DIR'),('SWM_HOST_RENDER','SWM_HOST_VALUE'),('SWM_PORT_RENDER','SWM_PORT_VALUE'),('WATCHDOG_DASH_URL_RENDER','WATCHDOG_DASH_URL_VALUE'),('WATCHDOG_STALE_SEC_RENDER','WATCHDOG_STALE_SEC_VALUE')]:
                shell+=target+'="$(xml_sed_escape "$'+key+'")"\n'
            shell+='ORG_PREFIX_RENDER="$(xml_sed_escape "com.$ORG.solarsindy")"\n'
            shell+='\n'.join('export '+key+'='+shell_quote(value) for key,value in trial_values.items())+'\n'
            shell+='for svc in monitor dashboard watchdog collector; do render "$svc"; done\n'
            script=directory/'render-extracted.bash';script.write_text(shell,encoding='utf-8',newline='\n')
            result=subprocess.run([args.bash,str(script).replace('\\','/')],env=baseline_env,capture_output=True,timeout=30)
            (directory/'stdout.log').write_bytes(result.stdout);(directory/'stderr.log').write_bytes(result.stderr)
            check(result.returncode==0,name+': actual render exits0')
            if result.returncode:continue
            check((work/'lint.calls').read_text().splitlines()==['lint']*4,name+': all4 render lint boundaries reached')
            check(not (work/'launchctl.called').exists(),name+': no service action')
            for service,suffix in fixed['suffix'].items():
                file=installed/('com.trial-render.solarsindy.'+suffix+'.plist')
                parsed=plistlib.loads(file.read_bytes())
                expected=dict(fixed['baseline_environment'][service])
                if service in fixed['consuming_services']:expected.update(trial_values)
                check(parsed['EnvironmentVariables']==expected,name+': '+service+' exact environment')
                check(parsed['ProgramArguments']==fixed['program_arguments'][service],name+': '+service+' exact escaped arguments')
                check(parsed['Label']=='com.trial-render.solarsindy.'+suffix,name+': '+service+' exact label')
                check('__EXTRA_ENV__' not in file.read_text(encoding='utf-8'),name+': '+service+' no remaining marker')
        # Execute the actual known-key diagnostic fragment without installer top level.
        for selected,expects_warning in [('monitor dashboard',False),('watchdog collector',True)]:
            name='diagnostic-'+selected.replace(' ','-');directory=output/name;directory.mkdir()
            shell='set -euo pipefail\n'+functions[2]+'\n'+known+'\nSERVICES=('+selected+')\n'
            shell+='\n'.join('export '+key+'='+shell_quote(value) for key,value in fixed['trial_values'].items())+'\n'+diagnostic+'\n'
            script=directory/'diagnostic-extracted.bash';script.write_text(shell,encoding='utf-8',newline='\n')
            result=subprocess.run([args.bash,str(script).replace('\\','/')],env=baseline_env,capture_output=True,timeout=30)
            (directory/'stdout.log').write_bytes(result.stdout);(directory/'stderr.log').write_bytes(result.stderr)
            check(result.returncode==0,name+': actual diagnostic exits0')
            stderr=result.stderr.decode('utf-8')
            for key in keys:check(('warning:   '+key in stderr)==expects_warning,name+': '+key+' diagnostic scope')
        record['complete']=all(c['passed'] for c in record['checks']) and len(record['checks'])==122
        check(len(record['checks'])==122,'exact independent check count')
    except BaseException as error:record.update(error_type=type(error).__name__,error=str(error))
    finally:
        record['failed']=[c['name'] for c in record['checks'] if not c['passed']]
        record['complete']=record['complete'] and not record['failed']
        with (output/'receipt.json').open('x',encoding='utf-8') as f:json.dump(record,f,indent=2)
        print(json.dumps(dict(complete=record['complete'],checks=len(record['checks']),failed=record['failed'],error=record.get('error'))))
    return 0 if record['complete'] else 1
if __name__=='__main__':raise SystemExit(main())
