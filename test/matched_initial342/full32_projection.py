"""Owned, bounded full32 generated tests; never writes the caller's Git tree.

Only Project/Manifest working-tree resolution deltas may be projected. Every
other tracked byte, original HEAD/index and the actual resolved environment are
rechecked. All forecasting, fitting, auditing and numerical assertions are Julia.
"""
from pathlib import Path, PurePosixPath
import argparse, copy, hashlib, importlib.util, json, os, re, shutil, signal, stat
import subprocess, sys, threading, time, tomllib, traceback

sys.dont_write_bytecode = True
OWNER_MEMBER = 'research/dbdt_completed_source/candidate_native_owner.py'
OWNER_SHA = '3c1a2838a3381afe22ae69b95afc1d3b03523b569f34034604390a7122cccac1'
LEGACY_MEMBER = 'research/dbdt_completed_source/native_stage.py'
LEGACY_SHA = '8799458af4fe532c62726df503bb88a18385fe670485363290e45447737cc836'
RAM, DISK = 8 * 1024**3, 16 * 1024**3
FILE_LIMIT, TREE_LIMIT = 64 * 1024**2, 512 * 1024**2
META_LIMIT, DIAGNOSTIC_LIMIT = 2 * 1024**2, 8 * 1024**2
SECONDS, SOURCE_SECONDS = 7200, 900
RESOLVER_MEMBERS = {'Project.toml', 'Manifest.toml'}
BATCH_TARGET = 8 * 1024**2


class ControllerTerminated(BaseException):
    """Unwind the fixed owner's finally before the POSIX controller exits."""


def install_termination_handler():
    if os.name != 'posix':
        return None
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        # A second soft signal must not interrupt owned-child cleanup. The
        # Julia parent's separate bounded hard escalation remains available.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise ControllerTerminated('POSIX controller received SIGTERM')
    signal.signal(signal.SIGTERM, terminate)
    return previous


def need(value, message):
    if not value:
        raise RuntimeError(message)


def ordinary(path, directory=False):
    path = Path(path)
    need(path.is_absolute(), 'absolute ordinary path required')
    need(path == Path(os.path.abspath(path)), 'noncanonical path spelling refused')
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        info = current.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not (getattr(info, 'st_file_attributes', 0) & 0x400),
             'symlink/reparse component refused: ' + str(current))
    info = path.stat()
    need(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode), 'ordinary path type differs')
    return path


def member(name):
    need(isinstance(name, str) and name and '\\' not in name and not any(c in name for c in '\0\r\n\t'), 'invalid source name')
    parsed = PurePosixPath(name)
    need(not parsed.is_absolute() and all(p not in ('', '.', '..', '.git') for p in name.split('/')),
         'source name escapes its boundary')
    return name


def digest(path, limit=FILE_LIMIT, deadline=None):
    ordinary(path)
    size = path.stat().st_size
    need(0 <= size <= limit, 'file byte bound exceeded: ' + str(path))
    h = hashlib.sha256()
    count = 0
    with path.open('rb') as stream:
        while True:
            if deadline is not None:
                need(time.monotonic() < deadline, 'whole-work deadline exhausted during file read')
            block = stream.read(65536)
            if not block:
                break
            count += len(block)
            need(count <= size, 'file grew during bounded read')
            h.update(block)
    need(count == size == path.stat().st_size, 'file changed size during bounded read')
    return h.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False) + '\n').encode('utf8')


def write_new(path, value):
    body = canonical(value)
    need(len(body) <= META_LIMIT, 'metadata byte bound exceeded')
    with path.open('xb') as stream:
        need(stream.write(body) == len(body), 'short metadata write')
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(body).hexdigest()


def read_json(path):
    body = bounded_bytes(path, META_LIMIT)
    def pairs(rows):
        out = {}
        for key, value in rows:
            need(key not in out, 'duplicate metadata key')
            out[key] = value
        return out
    return json.loads(body, object_pairs_hook=pairs)


def bounded_bytes(path, limit):
    ordinary(path)
    size = path.stat().st_size
    need(0 <= size <= limit, 'bounded reader file size differs')
    with path.open('rb') as stream:
        body = stream.read(limit + 1)
    need(len(body) == size == path.stat().st_size and len(body) <= limit, 'bounded reader file grew/changed size')
    return body


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def capture_child(request_path):
    """Runs inside the fixed owner's boundary; descendants inherit that boundary."""
    request = read_json(Path(request_path))
    overflow = threading.Event()
    failed = threading.Event()
    sizes = {}
    def drain(pipe, path, limit, key):
        total = 0
        try:
            with Path(path).open('xb', buffering=0) as stream:
                while True:
                    block = pipe.read(min(65536, limit + 1 - total))
                    if not block:
                        break
                    keep = block[:limit - total]
                    if keep:
                        need(stream.write(keep) == len(keep), 'short bounded capture write')
                    total += len(block)
                    if total > limit:
                        overflow.set()
                        break
                stream.flush()
        except BaseException:
            failed.set()
        finally:
            sizes[key] = min(total, limit)
            pipe.close()
    stdin = open(request['stdin_path'], 'rb') if request['stdin_path'] is not None else subprocess.DEVNULL
    try:
        process = subprocess.Popen(request['argv'], cwd=request['cwd'], stdin=stdin,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=dict(os.environ))
        write_new(Path(request['launch']), {'pid': process.pid, 'capture_pid': os.getpid()})
        threads = [threading.Thread(target=drain, args=(process.stdout, request['stdout'], request['limit'], 'stdout'), daemon=True),
                   threading.Thread(target=drain, args=(process.stderr, request['stderr'], DIAGNOSTIC_LIMIT, 'stderr'), daemon=True)]
        for thread in threads:
            thread.start()
        while process.poll() is None:
            if overflow.is_set() or failed.is_set():
                os._exit(86 if overflow.is_set() else 88)
            time.sleep(.01)
        for thread in threads:
            thread.join(5)
        if overflow.is_set() or failed.is_set() or any(t.is_alive() for t in threads):
            os._exit(86 if overflow.is_set() else 88)
        write_new(Path(request['result']), {'native_exit': process.returncode, 'pid': process.pid, 'capture_pid': os.getpid(), 'bytes': sizes})
        return 0
    finally:
        if stdin is not subprocess.DEVNULL:
            stdin.close()


def git_environment():
    allowed = {'GIT_PAGER': 'cat', 'GIT_TERMINAL_PROMPT': '0'}
    for key, value in os.environ.items():
        if key.startswith('GIT_'):
            need(allowed.get(key) == value, 'inherited Git override refused: ' + key)
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0')
    return env


class Controller:
    def __init__(self, source, out):
        self.source, self.out = source, out
        self.started = time.monotonic()
        self.deadline = self.started + SECONDS
        self.source_deadline = self.started + SOURCE_SECONDS
        self.source_phase = True
        self.count = 0
        self.owner = load_module(source / OWNER_MEMBER, 'matched342_fixed_owner')

    def remaining(self, deadline=None):
        value = min(self.deadline, deadline or self.deadline) - time.monotonic()
        need(value > 0, 'bounded full32 whole-work deadline exhausted')
        return value

    def command(self, argv, cwd, *, limit=FILE_LIMIT, stdin_path=None, seconds=60, retain=False, environment=None):
        self.count += 1
        folder = self.out / 'captures' / str(self.count)
        folder.mkdir()
        paths = {key: folder / (key + '.json' if key in ('launch', 'result', 'request') else key + '.bin')
                 for key in ('launch', 'result', 'request', 'stdout', 'stderr')}
        # The selected environment is inherited through the owner, never saved
        # in retained metadata (CI environments can contain credentials).
        request = dict(argv=list(map(str, argv)), cwd=str(cwd),
                       stdin_path=str(stdin_path) if stdin_path else None, limit=limit,
                       **{k: str(paths[k]) for k in ('launch', 'result', 'stdout', 'stderr')})
        write_new(paths['request'], request)
        launched = []
        def on_launch(pid):
            launched.append(pid)
            write_new(folder / 'owned-launch.json', {'owner_pid': os.getpid(), 'owned_leader_pid': pid})
        wall = min(seconds, self.remaining(self.source_deadline if self.source_phase else None))
        native = self.owner.run_owned([sys.executable, '-I', '-B', str(Path(__file__)), '--capture', str(paths['request'])],
            cwd=str(cwd), environment=environment or git_environment(), logpath=folder / 'owner.log', timeout=wall, on_launch=on_launch)
        write_new(folder / 'owner-result.json', native)
        need(len(launched) == 1 and native['pid'] == launched[0] and native['native_exit'] == 0 and not native['timed_out']
             and native['leader_reaped'] is True and native['ownership_boundary_closed'] is True
             and native['all_descendants_os_reaped'] is False, 'owned command failed; retained ' + str(folder))
        self.remaining()
        result = read_json(paths['result'])
        launch = read_json(paths['launch'])
        need(result['pid'] == launch['pid'] and result['capture_pid'] == launch['capture_pid'] == native['pid'], 'native/capture PID differs')
        need(result['native_exit'] == 0, 'native command failed; retained ' + str(folder))
        output = bounded_bytes(paths['stdout'], limit)
        need(len(output) <= limit and paths['stderr'].stat().st_size <= DIAGNOSTIC_LIMIT, 'capture byte bound differs')
        if retain:
            return output, native, launch, folder
        need(ordinary(folder, True).resolve() == folder and folder.parent == self.out / 'captures'
             and folder.name == str(self.count), 'owned capture cleanup boundary differs')
        shutil.rmtree(folder)
        return output

    def git(self, root, *arguments, limit=FILE_LIMIT, stdin_path=None):
        return self.command(['git', '-c', 'core.fsmonitor=false', '-C', str(root), *arguments], root, limit=limit, stdin_path=stdin_path)

    def text(self, root, *arguments):
        return self.git(root, *arguments, limit=4 * 1024**2).decode('utf8').strip()


def entries(raw, tree=False):
    result = {}
    for row in raw.decode('utf8').split('\0'):
        if not row:
            continue
        header, name = row.split('\t', 1)
        fields = header.split(' ')
        mode, oid = (fields[0], fields[2]) if tree else fields[:2]
        need(len(fields) == 3 and (fields[1] == 'blob' if tree else fields[2] == '0'), 'regular source index/tree required')
        member(name)
        need(mode in ('100644', '100755') and re.fullmatch('[0-9a-f]{40}', oid) and name not in result, 'source mode/OID differs')
        result[name] = (mode, oid)
    need(1 <= len(result) <= 8192 and len({n.casefold() for n in result}) == len(result), 'bounded unique portable source names required')
    return result


def batch_git(control, root, arguments, body, limit):
    """Use the same bounded native owner, retaining the exact finite stdin."""
    need(0 < len(body) <= META_LIMIT and 0 < limit <= FILE_LIMIT, 'batch transport bound exceeded')
    control.remaining(control.source_deadline)
    directory = control.out / 'batch-inputs'
    directory.mkdir(exist_ok=True)
    ordinary(directory, True)
    path = directory / (str(control.count + 1) + '.stdin')
    with path.open('xb') as stream:
        need(stream.write(body) == len(body), 'short batch input write')
        stream.flush()
        os.fsync(stream.fileno())
    result = control.git(root, *arguments, stdin_path=path, limit=limit)
    control.remaining(control.source_deadline)
    return result


def batch_object_sizes(control, root, names, index):
    body = ''.join(index[name][1] + '\n' for name in names).encode('ascii')
    output = batch_git(control, root, ('cat-file', '--batch-check'), body, META_LIMIT)
    lines = output.split(b'\n')
    need(lines[-1] == b'' and len(lines) == len(names) + 1, 'batch object count/framing differs')
    sizes = {}
    for name, line in zip(names, lines[:-1]):
        match = re.fullmatch(rb'([0-9a-f]{40}) blob (0|[1-9][0-9]*)', line)
        need(match is not None and match[1].decode('ascii') == index[name][1], 'batch object OID/type/order differs')
        size = int(match[2])
        need(size <= FILE_LIMIT, 'committed blob byte bound exceeded')
        sizes[name] = size
    return sizes


def batch_header(oid, rawsize, name=None):
    # Git's filtered batch header reports RAW object size. Its following
    # transformed payload is framed with the independently known working size.
    return (oid + ' blob ' + str(rawsize) + ('' if name is None else '\t' + name) + '\n').encode('utf8')


def batch_contents(output, rows, filtered=False):
    """Strict ordered frames, including binary LF/NUL; never split payloads."""
    cursor, result = 0, []
    view = memoryview(output)
    for name, oid, rawsize, size in rows:
        header = batch_header(oid, rawsize, name if filtered else None)
        need(output[cursor:cursor + len(header)] == header, 'batch content header/OID/size/path/order differs')
        cursor += len(header)
        end = cursor + size
        need(end < len(output) and output[end:end + 1] == b'\n', 'short/extra batch payload or delimiter differs')
        result.append((name, view[cursor:end]))
        cursor = end + 1
    need(cursor == len(output), 'extra batch records/bytes refused')
    return result


def verify_blob_batches(control, root, names, index, raw_sizes, hashes=None, sizes=None):
    filtered = hashes is not None
    pending, input_size, output_size = [], 0, 0

    def verify(name, body):
        if filtered:
            need(len(body) == sizes[name] and hashlib.sha256(body).hexdigest() == hashes[name],
                 'unchanged tracked checkout-filter bytes differ: ' + name)
        else:
            rawsize = raw_sizes[name]
            value = hashlib.sha1(('blob ' + str(rawsize) + '\0').encode('ascii'))
            value.update(body)
            need(len(body) == rawsize and value.hexdigest() == index[name][1], 'committed raw object integrity differs')

    def flush():
        nonlocal input_size, output_size
        if not pending:
            return
        body = ''.join(oid + (' ' + name if filtered else '') + '\n' for name, oid, _, _ in pending).encode('utf8')
        option = '--batch=%(objectname) %(objecttype) %(objectsize)\t%(rest)' if filtered else '--batch'
        arguments = ('cat-file', option, '--filters') if filtered else ('cat-file', option)
        output = batch_git(control, root, arguments, body, output_size)
        for name, content in batch_contents(output, pending, filtered):
            verify(name, content)
        control.remaining(control.source_deadline)
        pending.clear()
        input_size = output_size = 0

    for name in names:
        oid, rawsize = index[name][1], raw_sizes[name]
        size = sizes[name] if filtered else rawsize
        request_size = len((oid + (' ' + name if filtered else '') + '\n').encode('utf8'))
        frame_size = len(batch_header(oid, rawsize, name if filtered else None)) + size + 1
        # Preserve the old accepted payload cap and path semantics. Git's
        # whitespace batch grammar cannot retain a path's initial space.
        if frame_size > FILE_LIMIT or (filtered and name.startswith(' ')):
            flush()
            arguments = ('cat-file', '--filters', '--path=' + name, oid) if filtered else ('cat-file', 'blob', oid)
            verify(name, control.git(root, *arguments, limit=max(1, size)))
            control.remaining(control.source_deadline)
            continue
        if pending and (output_size + frame_size > BATCH_TARGET or input_size + request_size > META_LIMIT):
            flush()
        pending.append((name, oid, rawsize, size))
        input_size += request_size
        output_size += frame_size
        need(input_size <= META_LIMIT and output_size <= FILE_LIMIT, 'batch framing/input cap exceeded')
    flush()


def git_c_quote(raw):
    """Git C quoting is byte based; JSON's Unicode escapes are not supported."""
    need(isinstance(raw, bytes), 'byte input required for Git C quoting')
    quoted = bytearray(b'"')
    for value in raw:
        if value in (34, 92):
            quoted.extend((92, value))
        elif value < 32 or value >= 127:
            quoted.extend(('\\%03o' % value).encode('ascii'))
        else:
            quoted.append(value)
    quoted.append(34)
    return bytes(quoted)


def git_path_line(name):
    member(name)
    return git_c_quote(name.encode('utf8')) + b'\n'


def verify_clean_batches(control, root, names, index):
    pending, chunks, size = [], [], 0
    for name in names:
        # Git --stdin-paths consumes C-style quoted paths; UTF8 and embedded
        # quotes stay literal and the pathname supplies actual clean filters.
        line = git_path_line(name)
        need(len(line) <= META_LIMIT, 'batch pathname input cap exceeded')
        if pending and size + len(line) > META_LIMIT:
            chunks.append(pending)
            pending, size = [], 0
        pending.append((name, line))
        size += len(line)
    if pending:
        chunks.append(pending)
    for rows in chunks:
        output = batch_git(control, root, ('hash-object', '--stdin-paths'), b''.join(line for _, line in rows), META_LIMIT)
        lines = output.split(b'\n')
        need(lines[-1] == b'' and len(lines) == len(rows) + 1, 'clean batch result count/framing differs')
        for (name, _), line in zip(rows, lines[:-1]):
            need(re.fullmatch(rb'[0-9a-f]{40}', line) and line.decode('ascii') == index[name][1],
                 'unchanged tracked clean-filter OID differs: ' + name)
        control.remaining(control.source_deadline)


def snapshot(control, root, *, clean=False):
    ordinary(root, True)
    head = control.text(root, 'rev-parse', 'HEAD')
    tree = control.text(root, 'rev-parse', 'HEAD^{tree}')
    need(re.fullmatch('[0-9a-f]{40}', head) and re.fullmatch('[0-9a-f]{40}', tree), 'source HEAD/tree identity differs')
    index = entries(control.git(root, 'ls-files', '--stage', '-z'))
    need(index == entries(control.git(root, 'ls-tree', '-r', '-z', 'HEAD'), tree=True), 'source index differs from committed tree')
    need(not control.git(root, 'ls-files', '--others', '--exclude-standard', '-z'), 'untracked source/input refused')
    changed = {member(n) for n in control.git(root, 'diff', '--name-only', '-z', 'HEAD').decode('utf8').split('\0') if n}
    need(changed <= RESOLVER_MEMBERS and (not clean or not changed), 'only resolver Project/Manifest working-tree changes are permitted')
    status = control.git(root, 'status', '--porcelain=v1', '--untracked-files=all').decode('utf8')
    expected_status = ''.join(' M ' + n + '\n' for n in sorted(changed))
    need(status == expected_status, 'source status contains staged/renamed/deleted/untracked changes')
    git_controls = {}
    for name in ('HEAD', 'index', 'config', 'packed-refs'):
        path = Path(control.text(root, 'rev-parse', '--git-path', name))
        path = (root / path).resolve() if not path.is_absolute() else path
        if path.exists():
            git_controls[str(path)] = digest(path, deadline=control.source_deadline)
    hashes, sizes, total = {}, {}, 0
    for name, (mode, oid) in sorted(index.items()):
        path = ordinary(root / name)
        if os.name != 'nt':
            need(bool(path.stat().st_mode & 0o111) == (mode == '100755'), 'tracked executable mode changed: ' + name)
        size = path.stat().st_size
        total += size
        need(size <= FILE_LIMIT and total <= TREE_LIMIT, 'complete source byte bound exceeded')
        hashes[name] = digest(path, deadline=control.source_deadline)
        sizes[name] = size
    names = sorted(index)
    # Every member is requested and verified, including repeated OIDs and the
    # committed raw objects behind allowed resolver working-tree deltas.
    raw_sizes = batch_object_sizes(control, root, names, index)
    verify_blob_batches(control, root, names, index, raw_sizes)
    unchanged = [name for name in names if name not in changed]
    verify_clean_batches(control, root, unchanged, index)
    verify_blob_batches(control, root, unchanged, index, raw_sizes, hashes, sizes)
    for name in names:
        need(digest(root / name, deadline=control.source_deadline) == hashes[name], 'source bytes changed during proof')
    need(RESOLVER_MEMBERS <= hashes.keys(), 'resolved Project/Manifest missing')
    control.remaining(control.source_deadline)
    return dict(head=head, tree=tree, files_sha256=hashes, sizes=sizes,
                commit_object_ids={n: v[1] for n, v in sorted(index.items())}, modes={n: v[0] for n, v in sorted(index.items())},
                changed_paths=sorted(changed), status=status, git_controls=git_controls)


def recheck_original(control, root, before):
    # The complete initial raw-object/filter proof is retained above. Recheck
    # current raw bytes, exact HEAD/tree/index/status with no original writes.
    need(control.text(root, 'rev-parse', 'HEAD') == before['head'] and control.text(root, 'rev-parse', 'HEAD^{tree}') == before['tree'], 'original HEAD/tree changed')
    index = entries(control.git(root, 'ls-files', '--stage', '-z'))
    need({n: v[1] for n, v in index.items()} == before['commit_object_ids'] and {n: v[0] for n, v in index.items()} == before['modes'], 'original index changed')
    need(control.git(root, 'status', '--porcelain=v1', '--untracked-files=all').decode() == before['status'], 'original status changed')
    for name, pin in before['files_sha256'].items():
        need(digest(root / name, deadline=control.deadline) == pin, 'original source bytes changed: ' + name)
    for path, pin in before['git_controls'].items():
        need(digest(Path(path), deadline=control.deadline) == pin, 'original Git control bytes changed: ' + path)


def project_source(control, before):
    target = control.out / 'source'
    target.mkdir()
    for name, pin in sorted(before['files_sha256'].items()):
        source, dest = control.source / name, target / name
        ordinary(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        h = hashlib.sha256()
        with source.open('rb') as src, dest.open('xb') as dst:
            while True:
                control.remaining(control.source_deadline)
                block = src.read(65536)
                if not block:
                    break
                count += len(block)
                need(count <= before['sizes'][name], 'source grew during copy')
                need(dst.write(block) == len(block), 'short source copy')
                h.update(block)
        need(count == before['sizes'][name] and h.hexdigest() == pin and digest(source) == pin, 'copied source bytes differ')
        if os.name != 'nt':
            dest.chmod(0o755 if before['modes'][name] == '100755' else 0o644)
    control.git(target, '-c', 'init.templateDir=', 'init', '--template=', '--quiet')
    for key, value in (('core.autocrlf', 'false'), ('core.longpaths', 'true'), ('core.fsmonitor', 'false'), ('commit.gpgsign', 'false'),
                       ('user.name', 'SolarSINDy generated full32 tests'), ('user.email', 'generated-full32-tests@invalid')):
        control.git(target, 'config', '--local', key, value)
    hooks = control.out / 'empty-hooks'
    hooks.mkdir()
    control.git(target, 'config', '--local', 'core.hooksPath', str(hooks))
    control.git(target, 'add', '--force', '--all')
    index = entries(control.git(target, 'ls-files', '--stage', '-z'))
    need(set(index) == set(before['modes']), 'owned index changed the complete member set')
    mode_rows = ''.join(before['modes'][name] + ' ' + index[name][1] + '\t' + name + '\n'
                        for name in sorted(index) if before['modes'][name] != index[name][0])
    if mode_rows:
        modes_path = control.out / 'index-modes.txt'
        modes_path.write_bytes(mode_rows.encode('utf8'))
        control.git(target, 'update-index', '--index-info', stdin_path=modes_path)
    control.git(target, 'commit', '--no-gpg-sign', '--quiet', '-m', 'Record exact resolved source bytes for generated full32 tests')
    need(not control.git(target, 'remote'), 'owned generated source must have no publication remote')
    child = snapshot(control, target, clean=True)
    need(child['files_sha256'] == before['files_sha256'] and child['sizes'] == before['sizes'] and child['modes'] == before['modes'],
         'clean owned projection differs from tested source bytes/modes')
    return target, child


def supported_manifest(project, selected_manifest):
    """Check the actual manifest selected by the outside Julia driver."""
    project = ordinary(Path(project))
    selected = ordinary(Path(selected_manifest))
    plain = ordinary(project.parent / 'Manifest.toml')
    need(selected.resolve() == plain.resolve(),
         'only the actually selected same-directory ordinary Manifest.toml is supported')
    data = tomllib.loads(bounded_bytes(project, META_LIMIT).decode('utf8'))
    override = data.get('manifest')
    if override is not None:
        need(isinstance(override, str) and override and
             (project.parent / override).resolve() == plain.resolve(),
             'project manifest override changes the selected plain manifest')
    return plain


def unchanged_path_dependencies(manifest, original_root, projected_root):
    """A copied path token may not silently acquire a different referent."""
    tables = manifest.get('deps')
    need(isinstance(tables, dict), 'supported resolved manifest dependency tables required')
    for name, rows in tables.items():
        need(isinstance(rows, list), 'resolved dependency records must be arrays of tables')
        for row in rows:
            need(isinstance(row, dict), 'resolved dependency record must be a table')
            if name == 'SolarSINDy' or 'path' not in row:
                continue
            token = row['path']
            need(isinstance(token, str) and token, 'resolved dependency path token must be nonempty')
            old = (original_root / token).resolve()
            new = (projected_root / token).resolve()
            need(old == new, 'relocated non-SolarSINDy path dependency changes identity: ' + name)
            ordinary(old, True)


def environment_project(active, source, target, out, *, selected_manifest=None):
    """Preserve the actual resolved dependency graph; never invokes Pkg/resolve."""
    project = ordinary(active)
    # Actual main always passes Base's outside-observed selected manifest. The
    # explicit plain default also supports metadata-only engineering fixtures.
    manifest = supported_manifest(project, selected_manifest if selected_manifest is not None else active.parent / 'Manifest.toml')
    project_body, manifest_body = bounded_bytes(project, META_LIMIT), bounded_bytes(manifest, META_LIMIT)
    need(len(project_body) <= META_LIMIT and len(manifest_body) <= META_LIMIT, 'resolved environment byte bound exceeded')
    parsed = tomllib.loads(manifest_body.decode('utf8'))
    if source == target:
        unchanged_path_dependencies(parsed, manifest.parent, manifest.parent)
        return project, digest(project), digest(manifest), None
    if project.parent == source:
        need(project_body == bounded_bytes(target / 'Project.toml', META_LIMIT) and
             manifest_body == bounded_bytes(target / 'Manifest.toml', META_LIMIT), 'root resolved environment copy differs')
        supported_manifest(target / 'Project.toml', target / 'Manifest.toml')
        unchanged_path_dependencies(parsed, manifest.parent, target)
        return target / 'Project.toml', digest(project), digest(manifest), None
    rows = parsed.get('deps', {}).get('SolarSINDy')
    need(isinstance(rows, list) and len(rows) == 1 and rows[0].get('uuid') == 'e7a3c1d2-5f8b-4a9e-b6c0-3d2e1f4a8b7c'
         and (project.parent / rows[0].get('path', '')).resolve() == source, 'active resolved environment must name the original package path')
    # Replace exactly the path token in SolarSINDy's one existing table. Every
    # other byte and parsed dependency fact is preserved, including versions.
    text = manifest_body.decode('utf8')
    sections = list(re.finditer(r'^\[\[deps\.SolarSINDy\]\]\r?\n(?P<body>.*?)(?=^\[\[|\Z)', text, re.M | re.S))
    need(len(sections) == 1, 'one resolved SolarSINDy manifest table required')
    section = sections[0]
    matches = list(re.finditer(r'^path\s*=\s*("(?:[^"\\]|\\.)*")\s*$', section.group('body'), re.M))
    need(len(matches) == 1, 'one existing ordinary SolarSINDy path token required')
    match = matches[0]
    start, stop = section.start('body') + match.start(1), section.start('body') + match.end(1)
    replacement = json.dumps(str(target), ensure_ascii=False)
    newbody = (text[:start] + replacement + text[stop:]).encode('utf8')
    updated = tomllib.loads(newbody.decode('utf8'))
    expected = copy.deepcopy(parsed)
    expected['deps']['SolarSINDy'][0]['path'] = str(target)
    need(updated == expected, 'projection changed the actual resolved dependency graph')
    directory = out / 'resolved-environment'
    unchanged_path_dependencies(parsed, manifest.parent, directory)
    directory.mkdir()
    (directory / 'Project.toml').write_bytes(project_body)
    (directory / 'Manifest.toml').write_bytes(newbody)
    supported_manifest(directory / 'Project.toml', directory / 'Manifest.toml')
    return directory / 'Project.toml', hashlib.sha256(project_body).hexdigest(), hashlib.sha256(newbody).hexdigest(), dict(
        original_project_sha256=hashlib.sha256(project_body).hexdigest(), original_manifest_sha256=hashlib.sha256(manifest_body).hexdigest(),
        changed_package_uuid=rows[0]['uuid'], original_package_path=str(source), child_package_path=str(target),
        dependency_graph_equal_except_package_path=True)


def main(args):
    need(sys.version_info >= (3, 11) and not sys.flags.optimize, 'nonoptimized Python>=3.11 required')
    source = ordinary(args.package_root, True)
    out = args.destination
    need(out.is_absolute() and out == Path(os.path.abspath(out)), 'canonical absolute evidence path required')
    ordinary(out.parent, True)
    need(out.is_absolute() and not out.exists() and not out.is_symlink() and out != source
         and not out.is_relative_to(source) and not source.is_relative_to(out), 'fresh disjoint owned evidence directory required')
    julia = ordinary(args.julia)
    need(digest(source / OWNER_MEMBER) == OWNER_SHA and digest(source / LEGACY_MEMBER) == LEGACY_SHA, 'fixed native owner bytes differ')
    git_environment()
    out.mkdir()
    (out / 'captures').mkdir()
    control = Controller(source, out)
    previous_signal = install_termination_handler()
    try:
        selected = supported_manifest(args.active_project, args.selected_manifest)
        before = snapshot(control, source)
        write_new(out / 'ORIGINAL_SOURCE.json', before)
        # Always use a private projection. Besides resolver portability this
        # prevents Core's ordinary git-status cache refresh from touching even
        # the original clean index's cached-stat bytes.
        target, child = project_source(control, before)
        proof = dict(complete=True, head=child['head'], tree=child['tree'], source_files=len(child['files_sha256']),
                     source_sha256=child['files_sha256'], commit_object_ids=child['commit_object_ids'], all_clean_and_checkout_filters_match=True)
        proof_path = out / 'SOURCE_PROOF.json'
        proof_sha = write_new(proof_path, proof)
        active, active_sha, manifest_sha, environment_delta = environment_project(args.active_project, source, target, out,
            selected_manifest=selected)
        projection = dict(schema='matched342_generated_full32_source_projection_v1', original_root=str(source), child_root=str(target),
            original_head=before['head'], child_head=child['head'], child_tree=child['tree'], changed_paths=before['changed_paths'],
            complete_source_bytes_equal=True, source_proof_sha256=proof_sha, environment_delta=environment_delta,
            original_active_project=str(args.active_project), original_active_project_sha256=digest(args.active_project),
            original_selected_manifest=str(selected), original_active_manifest_sha256=digest(selected),
            projected_selected_manifest=str(active.parent / 'Manifest.toml'))
        projection_sha = write_new(out / 'SOURCE_PROJECTION.json', projection)
        recheck_original(control, source, before)
        # Admission is measured immediately before the real Julia child, after
        # projection/proof work. No RAM-floor skip or synthetic success exists.
        harness_path = source / 'tools/run_canonical_harness.py'
        need(digest(harness_path) == before['files_sha256']['tools/run_canonical_harness.py'], 'resource helper differs from tracked source')
        harness = load_module(harness_path, 'matched342_physical_memory')
        memory, method = harness.available_memory()
        disk = shutil.disk_usage(out).free
        need(type(memory) is int and memory >= RAM and type(disk) is int and disk >= DISK,
             '8GiB available physical memory and16GiB free disk are required before full32 Julia')
        generated = out / 'generated'
        generated.mkdir()
        result_path = out / 'JULIA_RESULT.json'
        depot = [os.path.abspath(p) for p in os.environ.get('JULIA_DEPOT_PATH', '').split(os.pathsep) if p]
        # Julia includes default system depot paths which need not yet exist.
        # Preserve this exact search list rather than inventing new depots.
        need(depot, 'actual resolved Julia depot must be supplied by parent')
        request = dict(schema='matched342_full32_owned_test_request_v1', package_root=str(target), active_project=str(active),
            active_project_sha256=active_sha, active_manifest_sha256=manifest_sha, depot_path=depot,
            source_proof=dict(path=str(proof_path), sha256=proof_sha), output_root=str(generated), result_path=str(result_path),
            original_head=before['head'], projected=target != source)
        request_path = out / 'REQUEST.json'
        request_sha = write_new(request_path, request)
        environment = git_environment()
        environment.pop('GIT_OPTIONAL_LOCKS', None)  # production Core rejects undeclared Git overrides
        environment.update(JULIA_NUM_THREADS='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', JULIA_LOAD_PATH='@' + os.pathsep + '@stdlib')
        argv = [str(julia), '--startup-file=no', '--threads=1', '--project=' + str(active.parent),
                str(target / 'test/matched_initial342/full32_worker.jl'), str(request_path), request_sha]
        control.source_phase = False
        _, native, launch, folder = control.command(argv, target, limit=FILE_LIMIT, seconds=control.remaining(), retain=True, environment=environment)
        worker = read_json(result_path)
        need(set(worker) == {'schema', 'complete', 'evidence_mode', 'julia_pid', 'cases', 'fit_count', 'nonlinear_complete', 'flat_complete',
                             'package_root', 'head', 'source_proof_sha256', 'active_project_sha256', 'active_manifest_sha256'},
             'closed full32 worker result fields differ')
        need(worker.get('schema') == 'matched342_full32_owned_test_worker_v1' and worker.get('complete') is True
             and worker.get('evidence_mode') == 'synthetic_only' and worker.get('julia_pid') == launch['pid']
             and worker.get('cases') == ['nonlinear', 'flat'] and worker.get('fit_count') == 64
             and worker.get('nonlinear_complete') is True and worker.get('flat_complete') is True
             and worker.get('head') == child['head'] and worker.get('source_proof_sha256') == proof_sha
             and worker.get('package_root') == str(target) and worker.get('active_project_sha256') == active_sha
             and worker.get('active_manifest_sha256') == manifest_sha,
             'both actual full32 generated cases must complete with exact source/native identity')
        output = bounded_bytes(folder / 'stdout.bin', FILE_LIMIT)
        need(output.count(b'MATCHED342_FULL32_AUDIT_B22_DONE flat=false') == 1 and
             output.count(b'MATCHED342_FULL32_AUDIT_B22_DONE flat=true') == 1, 'full32 completion marker count differs')
        recheck_original(control, source, before)
        need(digest(args.active_project) == projection['original_active_project_sha256'] and
             digest(args.active_project.parent / 'Manifest.toml') == projection['original_active_manifest_sha256'], 'original resolved environment changed')
        elapsed = time.monotonic() - control.started
        need(elapsed < SECONDS, 'full32 whole-work deadline exceeded')
        terminal = dict(schema='matched342_full32_owned_test_terminal_v1', complete=True, evidence_mode='synthetic_only',
            projected=target != source, original_checkout_unchanged=True, original_head=before['head'], child_head=child['head'],
            source_projection_sha256=projection_sha, source_proof_sha256=proof_sha, request_sha256=request_sha,
            owner_pid=os.getpid(), owned_leader_pid=native['pid'], julia_pid=launch['pid'], native=native, worker=worker,
            available_physical_bytes=memory, memory_method=method, free_disk_bytes=disk, wall_seconds=elapsed,
            stdout_sha256=digest(folder / 'stdout.bin'), stderr_sha256=digest(folder / 'stderr.bin'),
            worker_result_sha256=digest(result_path), individual_descendant_OS_reaping_observed=False)
        write_new(out / 'TERMINAL.json', terminal)
        print('Full32 nonlinear+flat generated producer/auditor/B22 verified; projection=' + str(target != source), flush=True)
        return 0
    except BaseException as error:
        # This directory was fresh and is owned by this invocation. A late
        # signal/output error cannot leave a success completion behind.
        terminal_path = out / 'TERMINAL.json'
        if terminal_path.exists():
            ordinary(terminal_path).unlink()
        failure = dict(schema='matched342_full32_owned_test_failure_v1', complete=False, error=type(error).__name__,
                       message=str(error)[:4000], owner_pid=os.getpid(), wall_seconds=time.monotonic() - control.started,
                       individual_descendant_OS_reaping_observed=False)
        write_new(out / 'FAILURE.json', failure)
        print('MATCHED342_FULL32_FAILURE_ARTIFACTS ' + str(out), file=sys.stderr, flush=True)
        traceback.print_exc(limit=10)
        return 1
    finally:
        if os.name == 'posix':
            signal.signal(signal.SIGTERM, previous_signal)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--capture':
        try:
            sys.exit(capture_child(sys.argv[2]))
        except BaseException as error:
            if isinstance(error, SystemExit):
                raise
            print(type(error).__name__ + ': ' + str(error)[:2000], file=sys.stderr, flush=True)
            os._exit(88)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--julia', type=Path, required=True)
    parser.add_argument('--active-project', type=Path, required=True)
    parser.add_argument('--selected-manifest', type=Path, required=True)
    sys.exit(main(parser.parse_args()))
