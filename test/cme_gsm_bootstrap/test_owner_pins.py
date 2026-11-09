"""Real clean-fixture source preflight and independent transitive pin oracles; no Julia."""
import ast
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
ENTRY='research/cme_gsm_bootstrap/run_native_bootstrap.py'
OWNER='research/dbdt_completed_source/native_stage.py'
MEMORY='tools/run_canonical_harness.py'
AUDIT='tools/canonical_dev_harness.sh'
STALE_MEMORY='3da7247d032942e43ec1fcf4243ac82c234f47848b6096ed88862be5068b6352'
OLD_OWNER='0b0f2327841230e5ac87d03bd67d1cbaf10717e0a52ce0134d71169a36146b89'
spec=importlib.util.spec_from_file_location('bootstrap_preflight_under_test',ROOT/ENTRY)
B=importlib.util.module_from_spec(spec);spec.loader.exec_module(B)

def digest(body):return hashlib.sha256(body).hexdigest()
def bounded(path):
    with path.open('rb') as stream:raw=stream.read(1024**2+1)
    if len(raw)>1024**2:raise ValueError('source fixture byte bound')
    return raw
def constant(body,name):
    for node in ast.parse(body).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('missing literal constant '+name)
def replace_constant(body,name,value):
    old=constant(body,name)
    if not isinstance(old,str) or body.count(old.encode())!=1:raise ValueError('ambiguous test mutation')
    return body.replace(old.encode(),value.encode(),1)

# These edges are independent literals from the frozen full-source AST/string
# map. They include second-order full-file consumers, not just native owners.
EDGES=(
 (ENTRY,OWNER,'OWNER_SHA'),(ENTRY,MEMORY,'MEMORY_SHA'),
 (MEMORY,OWNER,'OWNER_SHA'),(MEMORY,AUDIT,'AUDIT_SHA'),
 ('tools/run_full_scale_generated.py',OWNER,'OWNER_SHA'),
 ('tools/verify_canonical_harness_artifact.py',OWNER,'OWNER_SHA'),
 ('research/dbdt_completed_source/candidate_native_owner.py',OWNER,'LEGACY_SHA'),
 ('research/dbdt_completed_source/run_candidate_stage.py',OWNER,'LEGACY_OWNER_SHA'),
 ('research/dbdt_completed_source/run_candidate_stage.py','research/dbdt_completed_source/candidate_native_owner.py','OWNER_SHA'),
 ('test/candidate_stage_driver/run_controls.py','research/dbdt_completed_source/candidate_native_owner.py','PINS[OWNER]'),
 ('research/dbdt_completed_source/synthetic_checkpoint.py',OWNER,'file_sha(runner)'),
 ('research/dbdt_completed_source/run_synthetic_chain.py',OWNER,'file_sha(runner)'),
 ('research/dbdt_completed_source/candidate_validation_authority.jl',OWNER,'NATIVE_DRIVER_PINS[4]'),
 ('src/completed_source_dbdt/refit_artifacts/runtime.jl',OWNER,'_OWNED_RUNNER_SHA'),
)
def pin_value(body,selector):
    if selector=='PINS[OWNER]':
        tree=ast.parse(body)
        node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PINS' for t in n.targets))
        return next(ast.literal_eval(v) for k,v in zip(node.keys,node.values) if isinstance(k,ast.Name) and k.id=='OWNER')
    if selector=='file_sha(runner)':
        values=[ast.literal_eval(n.comparators[0]) for n in ast.walk(ast.parse(body))
                if isinstance(n,ast.Compare) and ast.unparse(n.left)=='file_sha(runner)'
                and len(n.comparators)==1 and isinstance(n.comparators[0],ast.Constant)]
        if len(values)!=1:raise ValueError('ambiguous runner pin')
        return values[0]
    if selector in ('NATIVE_DRIVER_PINS[4]','_OWNED_RUNNER_SHA'):
        import re
        pattern=rb'NATIVE_DRIVER_PINS\[4\]\s*==\s*"([0-9a-f]{64})"' if selector.startswith('NATIVE') else rb'const _OWNED_RUNNER_SHA\s*=\s*"([0-9a-f]{64})"'
        values=re.findall(pattern,body)
        if len(values)!=1:raise ValueError('ambiguous Julia pin')
        return values[0].decode()
    return constant(body,selector)
def graph_bodies():return {name:bounded(ROOT/name) for name in {p for edge in EDGES for p in edge[:2]}}
def verify_graph(bodies):
    if constant(bodies[ENTRY],'OWNER_RELATIVE')!=OWNER or constant(bodies[ENTRY],'MEMORY_RELATIVE')!=MEMORY:
        raise ValueError('fixed source cross-reference changed')
    for consumer,target,selector in EDGES:
        if pin_value(bodies[consumer],selector)!=digest(bodies[target]):
            raise ValueError('transitive pin differs: '+consumer+' -> '+target)

@contextmanager
def clean_fixture(*,root_characters=None):
    with tempfile.TemporaryDirectory(prefix='bootstrap-source-pin-control-') as temporary:
        base=Path(temporary).resolve(strict=True)
        if root_characters is None:
            root=base/'project'
        else:
            if type(root_characters) is not int or root_characters!=198:
                raise ValueError('fixed long-root geometry required')
            padding=root_characters-len(str(base))-len('/project')-1
            if not 1<=padding<=255:
                raise ValueError('temporary fixture base cannot fit the fixed198-character root')
            component='x'*padding
            parent=base/component;parent.mkdir()
            root=parent/'project'
            if len(str(root))!=198 or not root.is_relative_to(base):
                raise ValueError('long fixture escaped its fixed geometry')
        root.mkdir()
        empty_global=base/'empty-git-config';empty_global.write_bytes(b'')
        environment={key:value for key,value in os.environ.items() if not key.startswith('GIT_')}
        environment.update(GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=str(empty_global),GIT_TERMINAL_PROMPT='0')
        git=shutil.which('git')
        if git is None:raise RuntimeError('Git is required for the real source preflight regression')
        git=str(Path(git).resolve(strict=True))
        def command(*arguments):
            result=subprocess.run([git,'-C',str(root),*arguments],env=environment,check=True,capture_output=True,timeout=30)
            if len(result.stdout)>4*1024**2 or len(result.stderr)>4*1024**2:raise ValueError('Git capture bound')
            return result.stdout.decode().strip()
        for name in B.SOURCES:
            target=root/'research/cme_gsm_bootstrap'/name;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(bounded(ROOT/'research/cme_gsm_bootstrap'/name))
        for name in (OWNER,MEMORY):
            target=root/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(bounded(ROOT/name))
        command('init','--quiet');command('config','core.longpaths','false')
        if command('config','--local','--get','core.longpaths')!='false':
            raise ValueError('fixture local path configuration differs')
        command('config','user.name','Generated pin regression')
        command('config','user.email','pin-fixture@invalid');command('config','commit.gpgsign','false')
        hooks=base/'no-hooks';hooks.mkdir();command('config','core.hooksPath',str(hooks))
        def commit():
            command('add','--','research','tools');command('commit','--quiet','-m','Generated bounded source-pin fixture')
            return command('rev-parse','HEAD')
        head=commit()
        with mock.patch.dict(os.environ,environment,clear=True),mock.patch.object(B,'HERE',root/'research/cme_gsm_bootstrap'):
            yield root,head,commit

class BootstrapOwnerPinTests(unittest.TestCase):
    def test_current_full_transitive_pin_graph(self):
        verify_graph(graph_bodies())
        # Import fixed definitions only; __main__ and its native owner never run.
        spec=importlib.util.spec_from_file_location('bootstrap_summary_parser',Path(__file__).with_name('run_controls.py'))
        parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
        valid=b'Ran 9 tests in 0.001s\n\nOK\n'
        for raw in (valid,valid.replace(b'\n',b'\r\n')):
            with self.subTest(valid_summary=raw):parser.validate_summary(raw)
        malformed=(
            b'',valid.replace(b'9 tests',b'8 tests'),valid.replace(b'9 tests',b'09 tests'),
            valid.replace(b'Ran 9 tests',b'Ran nine tests'),
            valid.replace(b'Ran 9 tests in 0.001s\n',b'Ran 9 tests in 0.001s\nRan 9 tests in 0.002s\n'),
            valid+b'OK\n',valid.replace(b'OK\n',b'FAILED (failures=1)\n'),
            valid.replace(b'OK\n',b'OK (skipped=1)\n'),valid+b'ERROR: unexpected failure\n',
            valid.replace(b'OK\n',b'OK \n'),b'x'*(4*1024**2+1))
        for index,raw in enumerate(malformed):
            with self.subTest(malformed_summary=index):
                with self.assertRaises(RuntimeError):parser.validate_summary(raw)
    def test_stale_outer_helper_pin_is_rejected(self):
        bodies=graph_bodies();bodies[ENTRY]=replace_constant(bodies[ENTRY],'MEMORY_SHA',STALE_MEMORY)
        with self.assertRaises(ValueError):verify_graph(bodies)
    def test_coherent_outer_rehash_cannot_hide_stale_inner_owner_pin(self):
        bodies=graph_bodies();bodies[MEMORY]=replace_constant(bodies[MEMORY],'OWNER_SHA',OLD_OWNER)
        bodies[ENTRY]=replace_constant(bodies[ENTRY],'MEMORY_SHA',digest(bodies[MEMORY]))
        with self.assertRaisesRegex(ValueError,'transitive pin differs'):verify_graph(bodies)
    def test_fixed_relative_cross_reference_is_required(self):
        bodies=graph_bodies();bodies[ENTRY]=replace_constant(bodies[ENTRY],'MEMORY_RELATIVE',OWNER)
        with self.assertRaisesRegex(ValueError,'cross-reference'):verify_graph(bodies)
    def test_real_source_snapshot_accepts_clean_eight_file_fixture(self):
        with clean_fixture() as (root,head,_):
            snapshot=B.source_snapshot(root,head)
            self.assertEqual(len(snapshot),8)
            self.assertEqual(snapshot[OWNER],B.OWNER_SHA);self.assertEqual(snapshot[MEMORY],B.MEMORY_SHA)
        with self.subTest(root_characters=198):
            with clean_fixture(root_characters=198) as (root,head,_):
                names=tuple('research/cme_gsm_bootstrap/'+name for name in B.SOURCES)+(OWNER,MEMORY)
                target='research/cme_gsm_bootstrap/test_native_generated_artifact.jl'
                self.assertEqual(len(str(root)),198)
                self.assertEqual(max(len(str(root/name)) for name in names),259)
                self.assertTrue(all(len(str(root/name))<260 for name in names))
                expected=bounded(root/target)
                self.assertEqual(B.git(root,'cat-file','blob','HEAD:'+target),expected)
                self.assertEqual(B.git(root,'cat-file','blob',':'+target),expected)
                snapshot=B.source_snapshot(root,head)
                self.assertEqual(snapshot,{name:digest(bounded(root/name)) for name in names})
    def test_real_source_snapshot_rejects_original_stale_helper_pin(self):
        with clean_fixture() as (root,head,_),mock.patch.object(B,'MEMORY_SHA',STALE_MEMORY):
            with self.assertRaisesRegex(ValueError,'owner/memory helper pins differ'):B.source_snapshot(root,head)
    def test_real_snapshot_rejects_committed_helper_and_owner_mutations(self):
        for name in (MEMORY,OWNER):
            with self.subTest(member=name),clean_fixture() as (root,_,commit):
                with (root/name).open('ab') as stream:stream.write(b'\n# generated independent byte mutation\n')
                changed_head=commit()
                with self.assertRaisesRegex(ValueError,'owner/memory helper pins differ'):B.source_snapshot(root,changed_head)
    def test_real_snapshot_rejects_wrong_head_pin_and_dirty_source(self):
        for mode in ('head','pin','dirty'):
            with self.subTest(mode=mode),clean_fixture() as (root,head,_):
                if mode=='dirty':(root/'untracked-generated.txt').write_text('untracked',encoding='utf8')
                with mock.patch.object(B,'MEMORY_SHA','0'*64 if mode=='pin' else B.MEMORY_SHA):
                    with self.assertRaises((ValueError,subprocess.CalledProcessError)):
                        B.source_snapshot(root,'0'*40 if mode=='head' else head)
    def test_stale_main_preflight_cannot_load_or_launch_native_helpers(self):
        with clean_fixture() as (root,head,_),mock.patch.object(B,'MEMORY_SHA',STALE_MEMORY):
            out=root.parent/'negative-preflight'
            argv=['fixed-bootstrap-control','--project-root',str(root),'--evidence-dir',str(out),
                  '--runtime','1.12.6','--role','primary','--source-head',head]
            with mock.patch.object(sys,'argv',argv),mock.patch.object(B,'load_pinned',side_effect=AssertionError('native load forbidden')) as load:
                self.assertEqual(B.main(),1);load.assert_not_called()
            receipt=json.loads((out/'terminal.json').read_bytes())
            self.assertFalse(receipt['complete']);self.assertEqual(receipt['error_type'],'ValueError')
            self.assertFalse(receipt['source_authority_granted']);self.assertFalse(receipt['scientific_qualification'])
            self.assertNotIn('native_exit',receipt);self.assertFalse((out/'native.log').exists())

if __name__=='__main__':
    if sys.flags.optimize:raise RuntimeError('unoptimized Python assertions are required')
    unittest.main(verbosity=2)
