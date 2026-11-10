"""Fixed harmless portable-capture test child operations; no caller code evaluation."""
import sys

operation=sys.argv[1]
if operation=='complete_streams':
    import os
    os.write(1,b'complete expected stdout\n');os.write(2,b'complete expected stderr\n')
elif operation=='exact_boundary':
    import os
    os.write(1,b'a'*8388608);os.write(2,b'b'*8388608)
elif operation=='ordinary_failure':
    sys.stderr.write('ordinary failure');sys.exit(2)
elif operation=='overflow':
    import os
    os.write(1,b'a'*8388609);os.write(2,b'b'*8388609)
elif operation=='timeout_parent':
    import subprocess,time
    from pathlib import Path
    subprocess.Popen([sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'timeout_descendant',sys.argv[2]])
    time.sleep(60)
elif operation=='timeout_descendant':
    import os,time
    from pathlib import Path
    Path(sys.argv[2]).write_text(str(os.getpid()));time.sleep(60)
else:
    raise RuntimeError('unknown fixed portable-capture fixture operation')
