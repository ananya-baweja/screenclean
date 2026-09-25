"""Generate ``src/screenclean/render/corpus_en.txt``: short English lines for rendered test pages.

The lines look like what people photograph on screens: slide bullets, sentences, numbers,
prices, dates, times, e-mail addresses, URLs, code and error messages. Everything is built
from the word lists below, written for this project, so there is no personal data. Addresses
use the reserved ``example.com/.org/.net`` domains. Output is deterministic (fixed seed);
regenerate with::

    python tools/make_corpus.py
"""

from __future__ import annotations

import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "screenclean" / "render" / "corpus_en.txt"
SEED = 2026
N_LINES = 1500

SUBJECTS = [
    "The model",
    "Our team",
    "The new pipeline",
    "This dashboard",
    "The lecture",
    "The report",
    "Each student",
    "The server",
    "The survey",
    "The project",
    "The first experiment",
    "The quarterly review",
    "The camera",
    "The research group",
    "The library",
    "Every sensor",
    "The support desk",
    "The prototype",
    "The committee",
    "The weekly meeting",
    "The training run",
    "The database",
    "The mobile app",
    "The workshop",
    "The design",
]
VERBS = [
    "improves",
    "reduces",
    "measures",
    "explains",
    "compares",
    "tracks",
    "summarises",
    "doubles",
    "stores",
    "checks",
    "predicts",
    "updates",
    "records",
    "simplifies",
    "highlights",
    "filters",
    "exports",
    "reviews",
    "schedules",
    "estimates",
    "combines",
    "detects",
    "organises",
    "supports",
    "visualises",
]
OBJECTS = [
    "the error rate",
    "monthly revenue",
    "customer feedback",
    "the reading list",
    "response times",
    "energy usage",
    "the final grades",
    "network traffic",
    "image quality",
    "sales by region",
    "the project budget",
    "user sign-ups",
    "test coverage",
    "the delivery schedule",
    "memory usage",
    "the exam results",
    "weekly attendance",
    "the carbon footprint",
    "support tickets",
    "page load times",
    "the risk register",
    "survey responses",
    "the inventory",
    "lab safety rules",
    "the release notes",
]
ENDINGS = [
    "by twelve percent",
    "before the deadline",
    "across all regions",
    "for every course",
    "in real time",
    "without extra cost",
    "after each release",
    "during peak hours",
    "for the next quarter",
    "every morning",
    "with fewer errors",
    "on the shared drive",
    "for new users",
    "in the second term",
    "at the end of the week",
    "using open data",
    "since last year",
    "within two days",
    "on every device",
    "for the whole department",
]
TOPICS = [
    "Introduction",
    "Background",
    "Key findings",
    "Next steps",
    "Summary",
    "Methods",
    "Results",
    "Discussion",
    "Timeline",
    "Budget overview",
    "Open questions",
    "Lessons learned",
    "Action items",
    "Learning outcomes",
    "Risks and mitigations",
    "Project goals",
    "Data sources",
    "Evaluation plan",
    "Recommendations",
    "Case study",
    "Frequently asked questions",
    "Road map",
    "Team updates",
    "Agenda",
]
TITLE_THEMES = [
    "Signals and Systems",
    "Machine Learning Basics",
    "Quarterly Sales Review",
    "Cloud Migration Plan",
    "Data Structures",
    "Marketing Strategy",
    "Operating Systems",
    "Customer Success Report",
    "Linear Algebra",
    "Product Launch Checklist",
    "Computer Networks",
    "Annual Budget",
    "Digital Image Processing",
    "Research Methods",
    "Supply Chain Update",
    "Software Testing",
    "Statistics for Engineers",
    "Security Awareness",
    "Onboarding Guide",
    "Database Design",
    "Renewable Energy",
    "Team Retrospective",
]
BULLETS = [
    "Collect the raw data before Friday",
    "Review the draft with the whole team",
    "Book the lab for two hours",
    "Upload slides to the course page",
    "Back up the database every night",
    "Share the survey link by email",
    "Update the dependency versions",
    "Measure latency on a slow network",
    "Label one hundred sample images",
    "Check the results against the baseline",
    "Write unit tests for the parser",
    "Prepare a short demo video",
    "Rotate the access keys monthly",
    "Archive last year's reports",
    "Plan the next sprint on Monday",
    "Compare three pricing options",
    "Invite speakers for the seminar",
    "Document every design decision",
    "Fix the broken links on the website",
    "Reduce the image size before upload",
    "Print the handouts early",
    "Confirm the room booking",
    "Test the app on older phones",
    "Summarise feedback in one page",
]
ERRORS = [
    "Error: connection timed out after 30 seconds",
    "Warning: disk space below 10 percent",
    "Fatal: could not open file config.yaml",
    "TypeError: expected str, got NoneType",
    "Permission denied while writing to /var/log/app",
    "Build failed with exit code 2",
    "KeyError: 'user_id' not found in payload",
    "Server returned status 503 Service Unavailable",
    "ValueError: shape mismatch (3, 224, 224) vs (224, 224, 3)",
    "Invalid certificate for host api.example.com",
    "Out of memory: tried to allocate 512 MiB",
    "Syntax error near unexpected token 'fi'",
    "Checksum mismatch for package archive",
    "Login failed: session expired, please sign in again",
    "ImportError: cannot import name 'load_model'",
    "Timeout waiting for lock on table orders",
    "Unhandled exception in worker thread 4",
    "404 Not Found: /reports/2026/summary.pdf",
    "Deprecated: this option will be removed in version 3.0",
    "Retrying request (attempt 3 of 5)",
]
CODE = [
    "for i in range(len(items)):",
    "    total += items[i].price * items[i].qty",
    "return sorted(scores, reverse=True)",
    "def load_image(path: str) -> np.ndarray:",
    "import numpy as np",
    "model.eval()",
    "print(f'loss = {loss:.4f}')",
    "SELECT name, total FROM orders WHERE total > 100;",
    "UPDATE users SET active = 0 WHERE last_login < '2025-01-01';",
    "const result = await fetch(url);",
    "if (response.status !== 200) { throw new Error('failed'); }",
    'git commit -m "Fix typo in README"',
    "pip install -r requirements.txt",
    "docker run -p 8080:80 web:latest",
    "x = np.clip(x, 0.0, 1.0)",
    "with open('data.csv') as f:",
    "    rows = list(csv.reader(f))",
    "for (let i = 0; i < n; i++) {",
    "    sum += values[i];",
    "}",
    "while queue:",
    "    node = queue.pop(0)",
    "class Stack:",
    "    def push(self, item):",
    "        self.items.append(item)",
    "assert len(batch) == 32",
    "CREATE INDEX idx_date ON events (created_at);",
    "ls -la /home/project/data",
    "npm run build",
    "kubectl get pods --namespace staging",
    "optimizer.zero_grad()",
    "loss.backward()",
    "scheduler.step()",
    "df = pd.read_csv('sales.csv')",
    "df.groupby('region').sum()",
    "plt.savefig('figure.png', dpi=200)",
    "public static void main(String[] args) {",
    'System.out.println("Hello");',
    "int mid = (lo + hi) / 2;",
    "#include <stdio.h>",
    "return 0;",
    "try:",
    "except ValueError as err:",
    "    logger.warning(err)",
]
WORDS = [
    "team",
    "report",
    "data",
    "sales",
    "support",
    "info",
    "admin",
    "research",
    "events",
    "billing",
    "careers",
    "design",
    "library",
    "labs",
    "office",
    "orders",
    "press",
    "alerts",
    "backup",
    "course",
]
PATHS = [
    "docs",
    "reports",
    "courses",
    "api/v2",
    "help/faq",
    "blog",
    "downloads",
    "status",
    "pricing",
    "events",
    "learn/python",
    "products/tools",
    "support/tickets",
    "news/2026",
    "slides/week-3",
    "data/open",
]
MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
UNITS = ["kg", "km", "MB", "GB", "ms", "kWh", "units", "hours", "items", "users"]
CURRENCY = ["$", "EUR ", "INR ", "GBP "]


def sentence(r: random.Random) -> str:
    return f"{r.choice(SUBJECTS)} {r.choice(VERBS)} {r.choice(OBJECTS)} {r.choice(ENDINGS)}."


def question(r: random.Random) -> str:
    return f"How does {r.choice(SUBJECTS).lower()} affect {r.choice(OBJECTS)}?"


def title(r: random.Random) -> str:
    return f"{r.choice(TITLE_THEMES)}: {r.choice(TOPICS)}"


def number_line(r: random.Random) -> str:
    kind = r.randrange(6)
    if kind == 0:
        return f"Total: {r.choice(CURRENCY)}{r.randint(10, 99999):,}.{r.randint(0, 99):02d}"
    if kind == 1:
        return f"{r.choice(OBJECTS).capitalize()} rose {r.uniform(0.5, 45):.1f}% year on year"
    if kind == 2:
        return f"Order #{r.randint(10000, 99999)} - {r.randint(1, 250)} {r.choice(UNITS)}"
    if kind == 3:
        return (
            f"Accuracy {r.uniform(60, 99.9):.2f}%, loss {r.uniform(0.01, 2):.3f}, epoch {r.randint(1, 150)}"
        )
    if kind == 4:
        return f"Room {r.randint(100, 950)}, Block {r.choice('ABCDEFGH')}, Floor {r.randint(1, 12)}"
    return f"Phone: +1 555 {r.randint(100, 999)} {r.randint(1000, 9999)}"  # 555: reserved for fiction


def date_line(r: random.Random) -> str:
    y, m, d = r.randint(2019, 2027), r.randint(1, 12), r.randint(1, 28)
    kind = r.randrange(5)
    if kind == 0:
        return f"Deadline: {d} {MONTHS[m - 1]} {y}"
    if kind == 1:
        return (
            f"Meeting on {y}-{m:02d}-{d:02d} at {r.randint(8, 18):02d}:{r.choice(['00', '15', '30', '45'])}"
        )
    if kind == 2:
        return f"Last updated {d:02d}/{m:02d}/{y}, {r.randint(1, 12)}:{r.randint(0, 59):02d} PM"
    if kind == 3:
        return f"Week {r.randint(1, 52)}: {r.choice(TOPICS)}"
    return f"Q{r.randint(1, 4)} {y} results due {MONTHS[m - 1]} {d}"


def email_line(r: random.Random) -> str:
    tld = r.choice(["com", "org", "net"])
    addr = f"{r.choice(WORDS)}{r.choice(['', '.', '-'])}{r.choice(WORDS)}@example.{tld}"
    return r.choice([f"Contact: {addr}", f"Send questions to {addr}", addr, f"Reply-To: {addr}"])


def url_line(r: random.Random) -> str:
    host = f"{r.choice(['www.', 'docs.', 'app.', ''])}example.{r.choice(['com', 'org', 'net'])}"
    url = f"https://{host}/{r.choice(PATHS)}"
    if r.random() < 0.4:
        url += f"?id={r.randint(1, 9999)}"
    return r.choice([url, f"More at {url}", f"Link: {url}"])


VARS = [
    "total",
    "count",
    "score",
    "batch",
    "items",
    "result",
    "config",
    "rows",
    "image",
    "labels",
    "users",
    "prices",
]
FUNCS = ["load", "parse", "resize", "compute", "normalize", "fetch", "save", "update", "validate", "split"]
TABLES = ["orders", "users", "events", "courses", "invoices", "sensors", "products", "grades"]
COLUMNS = ["id", "name", "total", "created_at", "region", "status", "score", "price", "email", "city"]
CMDS = [
    "python train.py --epochs {n}",
    "git checkout -b feature/{v}",
    "mkdir -p data/{v}/raw",
    "curl -s https://api.example.com/{v}?limit={n}",
    "export BATCH_SIZE={n}",
    "tail -n {n} logs/{v}.log",
    "conda activate {v}",
    "scp {v}.zip backup.example.org:/srv/{v}/",
]


def code_line(r: random.Random) -> str:
    """A code or shell line: a fixed snippet, or a template filled with varied names and numbers."""
    v, f, n = r.choice(VARS), r.choice(FUNCS), r.randint(2, 512)
    kind = r.randrange(9)
    if kind == 0:
        return r.choice(CODE)
    if kind == 1:
        return f"{v} = {f}_{r.choice(VARS)}({r.choice(VARS)}, size={n})"
    if kind == 2:
        return f"    if {v} > {n}:"
    if kind == 3:
        return f"def {f}_{v}({r.choice(VARS)}, limit={n}):"
    if kind == 4:
        c1, c2 = r.sample(COLUMNS, 2)
        return f"SELECT {c1}, {c2} FROM {r.choice(TABLES)} WHERE {c1} > {n} ORDER BY {c2};"
    if kind == 5:
        return r.choice(CMDS).format(n=n, v=v)
    if kind == 6:
        return f"const {v} = {f}({r.choice(VARS)}.length * {n});"
    if kind == 7:
        return f"    {v}.append({f}(x) for x in {r.choice(VARS)}[:{n}])"
    return f"assert {v}.shape[0] == {n}, 'unexpected {r.choice(VARS)} size'"


def make_lines(n: int = N_LINES, seed: int = SEED) -> list[str]:
    r = random.Random(seed)
    makers = [
        (sentence, 34),
        (question, 6),
        (title, 8),
        (lambda r: r.choice(BULLETS), 8),
        (number_line, 12),
        (date_line, 9),
        (email_line, 5),
        (url_line, 5),
        (code_line, 14),
        (lambda r: r.choice(ERRORS), 5),
    ]
    funcs, weights = zip(*makers, strict=True)
    lines, seen = [], set()
    while len(lines) < n:
        line = r.choices(funcs, weights)[0](r)
        # Short, repeated lines (like "}") may repeat; everything else stays unique.
        if line in seen and len(line) > 12:
            continue
        seen.add(line)
        lines.append(line)
    return lines


def main() -> None:
    lines = make_lines()
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {len(lines)} lines to {OUT.relative_to(OUT.parents[3])}")


if __name__ == "__main__":
    main()
