# Keeping the VoxLive bill small

Prices below are `ap-south-1` (Mumbai) on-demand list, rounded, and were
current when this was written. Treat them as ratios rather than quotes — the
ranking of the line items is stable, the absolute numbers are not. Check the
AWS pricing calculator before committing to anything.

---

## 1. The bill is not mostly AWS

Every second of retained speech becomes a Vertex AI call. At any realistic
usage level, **Gemini is the largest line on the combined bill, and AWS is the
second**. Two settings in this repo drive Gemini spend directly and neither is
an AWS concern:

| Setting | Effect on Gemini cost |
|---|---|
| `CONTEXT_SEGMENTS=4` | ~1,200 characters of extra input prepended to *every* call |
| `_split()` in `state.py` | one segment cut at speaker changes becomes 2–3 separate calls |
| `SOFT_MAX_SEGMENT_MS=3500` | shorter segments = more calls for the same audio |

`SOFT_MAX_SEGMENT_MS` was lowered from 6000 to 3500 for latency, and it worked
— but it also increased the call count for a given minute of audio by roughly
70%. That is a real trade you have made, and it is worth measuring rather than
assuming: if per-call overhead dominates, raising it back toward 5000 costs
latency you can feel and saves money you can see.

Before optimising AWS, put a week of real usage through the system and read the
Vertex AI billing breakdown. Optimising a $30 compute bill while a $200 model
bill sits next to it is the wrong order.

---

## 2. Fix the container before anything else

`requirements.txt` pins `torch>=2.2`. Installed from PyPI on Linux, that is the
**CUDA build**: torch plus `nvidia-cudnn`, `nvidia-cublas`, `nvidia-nccl` and
the rest — roughly 2 GB of GPU libraries that cannot execute on Fargate,
because Fargate has no GPU.

| | installed size | image |
|---|---|---|
| default `pip install torch` | ~2.4 GB | ~3.2 GB |
| `--index-url .../whl/cpu` | ~0.2 GB | ~1.1 GB |

You pay that on every image pull: every deploy, every scale-out, every task
replacement. `deploy/Dockerfile` fixes it in one line. It also bakes the
WeSpeaker weights in at build time, so a cold start does not pull ~90 MB from
Hugging Face through your NAT gateway.

**Build for `linux/arm64` and run on Fargate Graviton.** Same vCPU and memory,
about 20% cheaper, and the CPU torch wheel exists for aarch64.

---

## 3. The three AWS line items that cost more than the compute

### NAT Gateway — ~$32/month before a byte moves

The default "tasks in private subnets" pattern needs one, because the backend
must reach Vertex AI outbound. For a single stateless service this buys very
little: put the tasks in **public subnets with `assignPublicIp: ENABLED`** and
a security group that allows no inbound except from the load balancer. A
public IP is not an exposure; an open security group is, and you control that
directly.

If you later need private subnets for RDS, keep the *tasks* public and the
*database* private. That is the split that matters.

### CloudWatch Logs retention — silently unbounded

`observability/logging.py` emits structured JSON per segment. At ~10 segments
per minute per session that is real volume, and **CloudWatch log groups default
to "Never expire"**. Ingestion is charged once (~$0.50/GB) and storage every
month thereafter, forever, for logs nobody will read.

Set retention to 14 or 30 days on the log group the day you create it. This is
the single most commonly forgotten recurring cost in an ECS deployment.

### Aurora Serverless v2 — do not

The minimum 0.5 ACU floor runs about $43/month for a database that will hold a
few hundred rows. `db.t4g.micro`, single-AZ, is roughly $12–15/month and is
more than this schema needs. Multi-AZ doubles it and buys failover you do not
yet have a use for.

---

## 4. Two shapes, honestly compared

### ECS Fargate + ALB + RDS — the shape the code is written for

| Item | Monthly |
|---|---|
| Fargate 1 task, 2 vCPU / 4 GB, ARM, always on | ~$50 |
| Application Load Balancer (needed for WSS + TLS) | ~$18 |
| RDS `db.t4g.micro` single-AZ + 20 GB | ~$15 |
| ECR storage, CloudWatch, data transfer | ~$8 |
| NAT Gateway *if* you use private subnets | +$32 |
| **Total** | **~$90, or ~$122 with NAT** |

### One EC2 instance running Docker Compose

| Item | Monthly |
|---|---|
| `t4g.medium` (2 vCPU / 4 GB, ARM) — 1-yr Compute Savings Plan | ~$18 |
| 30 GB gp3 root volume | ~$3 |
| Elastic IP (attached) | $0 |
| Caddy for automatic TLS, Postgres in a container | $0 |
| **Total** | **~$21** |

**For an internship-stage product with a handful of concurrent sessions, take
the second one.** It is roughly a quarter of the cost and it runs the same
container. What you give up is real and worth naming: no rolling deploys, no
automatic instance replacement, and Postgres backups become your job (a nightly
`pg_dump` to S3 with a lifecycle rule costs cents).

Move to ECS when you have a reason — a second engineer deploying, a customer
asking about availability, or measured load that one instance cannot hold — not
before. The code does not care: `main.py` and the Dockerfile are identical
either way.

**Do not run Fargate Spot for this.** Spot interruption gives two minutes'
notice and kills every live WebSocket on the task. A transcription session
disappearing mid-sentence is a much worse outcome than a slightly larger bill.

---

## 5. The quota gap that is a bill risk

`Plan.max_monthly_minutes` is defined in `tenant/models.py` and **checked
nowhere**. `AdmissionController` counts concurrent sessions only, which bounds
how many sessions run at once and says nothing about how many minutes they
consume. One tenant on the Starter plan can run five concurrent sessions
continuously for a month and generate an unbounded Vertex AI bill entirely
within their plan limits.

`Plan.max_session_minutes` had the same problem. That one is now enforced —
`SessionState.feed_audio` stops accepting audio at the ceiling, drains the
in-flight work so the transcript is complete, and closes with 1013 — which
caps a single forgotten browser tab.

The monthly ceiling still needs the sessions table to be real, because it
requires summing completed durations for the organization in the current
month. The partial index for exactly that query is already in
`docs_schema.sql`:

```sql
CREATE INDEX sessions_org_completed_idx ON sessions (organization_id, ended_at)
    WHERE status = 'completed';
```

Until that query exists, set a **billing alarm on the GCP project**. It is not
a substitute for a quota — it tells you after the money is spent — but it is
five minutes of work and it is the difference between noticing on the 3rd and
noticing on the 30th.

---

## 6. Checklist

- [ ] Build with the CPU torch wheel (`deploy/Dockerfile`)
- [ ] Build and run `linux/arm64`
- [ ] Set CloudWatch log retention to 14–30 days
- [ ] No NAT Gateway; tasks in public subnets, tight security group
- [ ] `db.t4g.micro` single-AZ, not Aurora Serverless
- [ ] ECR lifecycle policy: keep the last 5 images
- [ ] Billing alarm on the GCP project *and* an AWS Budget at your expected spend
- [ ] `DIARIZATION_MODE=off` for any tenant that does not need speakers — it is
      the only reason the task needs 4 GB instead of 1 GB
- [ ] ALB access logs must not record query strings — the auth token is in
      the WebSocket URL
