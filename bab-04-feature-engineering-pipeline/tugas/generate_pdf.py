import os
import subprocess
from pathlib import Path

tugas_dir = Path(r"d:\Anggit Arisa Siwi Pratama\Unindra\Machine Learning\ml-workspace\3-ml-assignment\bab-04-feature-engineering-dan-pipeline\tugas")
html_path = tugas_dir / "laporan_praktikum_4_printable.html"
pdf_out = tugas_dir / "LAPORAN_PRAKTIKUM_4_ANGGIT_ARISA_SIWI_PRATAMA_202343570045.pdf"

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
if not os.path.exists(edge_path):
    edge_path = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

cmd = [
    edge_path,
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={str(pdf_out)}",
    str(html_path)
]

print("Executing Edge headless to generate PDF...")
res = subprocess.run(cmd, capture_output=True, text=True)
print("Return code:", res.returncode)

if pdf_out.exists():
    print(f"[SUCCESS] PDF generated: {pdf_out} ({pdf_out.stat().st_size} bytes)")
else:
    print("[ERROR] PDF was not generated.")
