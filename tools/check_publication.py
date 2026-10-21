"""Scan tracked source and Git index without printing sensitive values."""
from pathlib import Path
import json, re, subprocess
ROOT = Path(__file__).resolve().parents[1]
SKIP = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist'}
RULES = {
 'private-key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
 'github-token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
 'aws-key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
 'literal-secret': rb'(?i)(?:api[_-]?key|api[_-]?secret|password|access[_-]?token)\s*[=:]\s*[\x22\x27]([A-Za-z0-9_+/=-]{24,})[\x22\x27]',
 'personal-path': rb'(?i)[A-Z]:[/\\]+Users[/\\]+(?!Public[/\\]|Default[/\\]|<)[A-Za-z0-9_.-]+[/\\]',
}
def main():
    git = subprocess.run(['git', '-C', str(ROOT), 'ls-files', '-z', '--cached'], capture_output=True)
    indexed = git.returncode == 0 and (ROOT / '.git').exists()
    names = [n.decode('utf-8') for n in git.stdout.split(b'\0') if n] if indexed else [p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and not set(p.relative_to(ROOT).parts) & SKIP]
    findings=[]
    for name in names:
        path=Path(name)
        if (set(path.parts) & (SKIP | {'data', 'outputs', '.tools', 'backups'}) or (path.name.startswith('.env') and path.name != '.env.example') or path.suffix in {'.pem','.key','.db','.sqlite','.log','.exe','.zip','.pyc'}):
            findings.append({'path':name,'rule':'private-or-generated-file'});continue
        samples=[]
        if indexed:
            result=subprocess.run(['git','-C',str(ROOT),'show',':'+name],capture_output=True)
            if result.returncode: findings.append({'path':name,'rule':'index-unreadable'})
            else:samples.append(result.stdout)
        if (ROOT/name).is_file():samples.append((ROOT/name).read_bytes())
        for rule,pattern in RULES.items():
            if any(re.search(pattern,s) for s in samples): findings.append({'path':name,'rule':rule})
    print(json.dumps({'passed':not findings,'files_checked':len(names),'scope':'index and working source' if indexed else 'working source','findings':findings},ensure_ascii=False,indent=2))
    return int(bool(findings))
if __name__=='__main__': raise SystemExit(main())
