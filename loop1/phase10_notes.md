# Phase 10 Notes — Specialist Routing UI + Doctor Portal

## Context

Phase 9 built the backend routing engine (`src/loop3/routing/router.py`) that
takes the final differential and returns a `RoutingDecision` — a specialty,
urgency level, and appointment type. Phase 10 wires that engine into the web
app and adds two new user-facing features:

1. **Patient side**: After session ends, show the AI-recommended specialist and
   a list of bookable doctors in that specialty. Patient can book a time slot.

2. **Doctor portal**: A separate, doctor-authenticated page where doctors see
   their appointment queue, full AI session transcript, per-turn differential
   evolution, final diagnosis + confidence, and critic scores.

---

## What Already Exists

### Backend (don't touch these unless extending)

| File | What it does |
|---|---|
| `web/api.py` | FastAPI app — all routes live here |
| `web/api_session.py` | `APISession` class — drives one diagnostic session |
| `src/loop1/` | Doctor LLM, compression, retrieval, schemas |
| `src/loop2/critic/` | Critic model, aggregator |
| `src/loop3/routing/` | Phase 9 routing engine |
| `data/disease_specialty_map.json` | Disease → specialty + urgency mapping |

### Frontend (files in `web/static/`)

| File | What it does |
|---|---|
| `login.html` | Patient sign-in / register |
| `index.html` | Case select grid (6 predefined cases) |
| `patient-info.html` | Custom complaint intake form |
| `session.html` | Live diagnostic dialogue view |
| `report.html` | Final session report with critic scores |
| `my-sessions.html` | Patient's session history |
| `history.html` | (legacy) |
| `styles.css` | Global dark-theme CSS |
| `auth.js` | JWT token storage and auth helpers |

### Existing API endpoints in `web/api.py`

```
GET  /api/cases                      → list predefined case cards
GET  /api/cases/{case_id}            → single case card
POST /api/session/start              → start session from predefined case
POST /api/session/start-custom       → start session from custom complaint
POST /api/session/{id}/turn          → submit one patient answer
GET  /api/session/{id}/report        → final report JSON
POST /api/register                   → create patient account
POST /api/login                      → patient login
GET  /api/me                         → current user info
GET  /api/sessions                   → patient session history
```

---

## What Phase 10 Needs to Build

### Part A — Specialist Recommendation (patient-facing)

**Flow**: Session ends → `report.html` calls new `/api/session/{id}/routing`
endpoint → display recommended specialty + doctor cards → patient clicks Book →
slot is saved as an appointment.

#### A1. Mock doctor directory

Create `web/data/doctors.json` — a static list of mock doctors. No real
database needed for the prototype.

```json
[
  {
    "id": "dr_001",
    "name": "Dr. Sarah Chen",
    "specialty": "Cardiology",
    "hospital": "City Heart Centre",
    "rating": 4.8,
    "avatar_initials": "SC",
    "available_slots": [
      "2026-06-17T09:00",
      "2026-06-17T14:00",
      "2026-06-18T10:30"
    ]
  },
  ...
]
```

Cover these specialties (2–3 doctors each):
Cardiology, Pulmonology, Neurology, Gastroenterology, Internal Medicine,
Infectious Disease, Rheumatology, Hematology, Emergency Medicine,
Endocrinology, Nephrology, General Surgery.

For Emergency Medicine entries, note in the card that the patient should go
to the nearest ED — no booking slot needed.

#### A2. New API endpoints

```
GET  /api/session/{id}/routing       → returns RoutingDecision + matching doctors
POST /api/session/{id}/book          → patient books a slot
GET  /api/appointments               → (auth required) patient's booked appointments
```

**`GET /api/session/{id}/routing` response shape:**

```json
{
  "specialty": "Cardiology",
  "urgency": "urgent",
  "appointment_type": "urgent_appointment_within_48h",
  "reasoning": "Possible NSTEMI requires immediate cardiac evaluation",
  "is_emergency": false,
  "doctors": [
    {
      "id": "dr_001",
      "name": "Dr. Sarah Chen",
      "hospital": "City Heart Centre",
      "rating": 4.8,
      "available_slots": ["2026-06-17T09:00", "2026-06-17T14:00"]
    }
  ]
}
```

**`POST /api/session/{id}/book` request body:**

```json
{
  "doctor_id": "dr_001",
  "slot": "2026-06-17T09:00"
}
```

Response: `{ "appointment_id": "appt_uuid", "confirmed": true }`

#### A3. Appointments store

Save booked appointments to `web/data/appointments.json`. Same pattern as
`web/data/users.json` — load/save on every write.

```json
{
  "appt_uuid": {
    "appointment_id": "appt_uuid",
    "session_id": "...",
    "patient_user_id": "...",
    "doctor_id": "dr_001",
    "doctor_name": "Dr. Sarah Chen",
    "specialty": "Cardiology",
    "slot": "2026-06-17T09:00",
    "booked_at": "2026-06-16T12:00:00Z",
    "primary_diagnosis": "Possible NSTEMI",
    "urgency": "urgent"
  }
}
```

#### A4. UI changes to `report.html`

After the existing final diagnosis section, add a **"Next Steps" block**:

```
┌─────────────────────────────────────────┐
│  Recommended Specialist                  │
│  ─────────────────────────────────────  │
│  Based on your diagnosis, we recommend  │
│  seeing a Cardiologist.                 │
│                                         │
│  [urgency badge: URGENT]                │
│  [reasoning text from RoutingDecision]  │
│                                         │
│  Available Cardiologists                │
│  ─────────────────────────────────────  │
│  [Doctor card] [Doctor card]            │
│  [Doctor card]                          │
│                                         │
│  Each card: name, hospital, rating,     │
│  "Book Appointment" → slot picker modal │
└─────────────────────────────────────────┘
```

Emergency routing: skip the doctor cards. Show a red alert:
"This requires emergency care. Please go to your nearest Emergency Department
or call 999 immediately."

After booking: replace the cards with a green confirmation:
"Appointment confirmed with Dr. Sarah Chen on 17 Jun at 09:00."

---

### Part B — Doctor Portal

Doctors have a separate account type. They log in at the same `/login.html`
page but are redirected to `doctor-portal.html` instead of `index.html`.

#### B1. Doctor accounts

Add a `role` field to the user schema in `web/data/users.json`:

```json
{
  "user_id_123": {
    "user_id": "user_id_123",
    "name": "Dr. Sarah Chen",
    "email": "sarah.chen@cityheart.com",
    "password_hash": "...",
    "role": "doctor",
    "doctor_id": "dr_001",
    "specialty": "Cardiology"
  }
}
```

Seed 3–4 doctor accounts at startup if they don't already exist (same pattern
as any seed data). Default password e.g. `Doctor123!` for each.

Update `_get_user_from_request()` — it already returns the user dict; just make
sure `role` is passed through to the response of `GET /api/me`.

Update `POST /api/login` to redirect hint: return `{ ..., "role": "doctor" }`
so the frontend can route to the right page.

#### B2. New doctor API endpoints

```
GET  /api/doctor/appointments              → list appointments assigned to this doctor
GET  /api/doctor/appointments/{appt_id}   → full detail: patient info + AI session data
```

Both require auth AND `role == "doctor"`. Return 403 if patient account tries.

**`GET /api/doctor/appointments` response:**

```json
[
  {
    "appointment_id": "appt_uuid",
    "patient_name": "Anonymous Patient",
    "slot": "2026-06-17T09:00",
    "primary_diagnosis": "Possible NSTEMI",
    "urgency": "urgent",
    "session_id": "...",
    "booked_at": "2026-06-16T12:00:00Z"
  }
]
```

**`GET /api/doctor/appointments/{appt_id}` response:**

```json
{
  "appointment": { ...full appointment record... },
  "session_report": { ...same shape as GET /api/session/{id}/report... },
  "routing": { ...RoutingDecision... }
}
```

This endpoint loads the session JSONL from disk (same as `get_report` does
already in `api.py:208 _load_report_from_disk()`) and attaches it.

#### B3. `doctor-portal.html` — new file

Three sections:

**1. Appointments queue** (left sidebar or top section)

```
┌──────────────────────────────────┐
│  Upcoming Appointments           │
│  ──────────────────────────────  │
│  [Patient] 17 Jun 09:00          │
│  Possible NSTEMI • URGENT        │
│  [View Details]                  │
│                                  │
│  [Patient] 18 Jun 14:00          │
│  Migraine • ROUTINE              │
│  [View Details]                  │
└──────────────────────────────────┘
```

**2. Session detail view** (main panel, shown when "View Details" clicked)

- Patient demographics (age, sex, chief complaint)
- **Differential evolution timeline**: turn-by-turn animated bars showing how
  the top diagnoses shifted across the conversation. One row per diagnosis,
  bars showing probability per turn.
- **Full conversation transcript**: doctor questions and patient answers in
  chronological order (same bubble style as `session.html`)
- **Final diagnosis block**: big callout with the top dx and confidence
- **Critic scores table**: one row per turn, Q-quality / Diff-quality /
  Reasoning scores as coloured badges
- **AI reasoning**: the `rationale` fields from each turn collapsed/expandable

**3. Quick stats header**

Total appointments today, average AI confidence, any emergency-level urgency
flagged in red.

---

## Routing Integration in `api.py`

The routing endpoint needs to call the Phase 9 router. The router lives at
`src/loop3/routing/router.py`. Import it the same way `api_session.py` imports
from `loop1` and `loop2`.

```python
from loop3.routing.router import route as compute_routing
from loop3.routing.routing_schema import RoutingDecision
```

The session's `_final_record.final_differential` is already a list of
`DiagnosisEntry` objects with `.dx` and `.prob`. Convert to the list of tuples
the router expects:

```python
differential = [(d.dx, d.prob) for d in session._final_record.final_differential]
confidence = session._final_record.final_differential[0].prob if differential else 0.0
routing = compute_routing(session_id, differential, confidence)
```

Then filter `doctors.json` by `routing.primary_routing.specialty` and return
the matching doctors in the response.

---

## Auth Guard Pattern (already used in `api.py`)

The existing pattern:

```python
user = _get_user_from_request(request)
if not user:
    raise HTTPException(status_code=401, detail="Not authenticated")
```

For doctor-only endpoints, add:

```python
if user.get("role") != "doctor":
    raise HTTPException(status_code=403, detail="Doctor account required")
```

---

## Build Order

### Step 1 — Doctor directory JSON
Create `web/data/doctors.json` with 2–3 doctors per specialty listed above.

### Step 2 — Appointments store
Create empty `web/data/appointments.json` (`{}`). Add load/save helpers in
`api.py` modelled on `_load_users()` / `_save_users()`.

### Step 3 — Routing endpoint
`GET /api/session/{id}/routing` — call the Phase 9 router, filter doctors.json,
return combined response.

### Step 4 — Booking endpoint
`POST /api/session/{id}/book` — validate slot exists on that doctor, write to
appointments.json, return confirmation.

### Step 5 — Patient appointments endpoint
`GET /api/appointments` — return all appointments for the authenticated patient.

### Step 6 — Report.html "Next Steps" block
Fetch routing on page load. Render specialty card + doctor cards. Wire up
booking modal. Show emergency alert if urgency == emergency.

### Step 7 — Doctor accounts
Add `role` and `doctor_id` to user schema. Seed doctor accounts. Update login
to return role. Update `GET /api/me` to include role.

### Step 8 — Doctor portal endpoints
`GET /api/doctor/appointments` and `GET /api/doctor/appointments/{id}`.

### Step 9 — `doctor-portal.html`
Appointments sidebar + session detail panel. Differential evolution chart (pure
CSS/JS bars, no chart library needed — just animate width transitions).

### Step 10 — Login routing
In `auth.js` / `login.html` — after successful login, check `role` in response
and redirect to `doctor-portal.html` if doctor, `index.html` if patient.

---

## Design Constraints

- **Same dark theme** — background `#0f1117`, surfaces `#1a1f2e`, accent teal
  `#00b4d8`, white/gray text. Match `styles.css` exactly.
- **No new JS libraries** — all existing pages use vanilla JS. Keep it that way.
- **No real payment or calendar** — booking just writes to `appointments.json`.
- **Patients don't see critic scores** — those are only in the doctor portal.
  The `report.html` "Next Steps" block shows routing + booking only.
- **No doctor self-registration** — doctor accounts are seeded in `api.py`
  at startup. Doctors cannot create accounts via the register form.

---

## Urgency Badge Colours

| Urgency | Badge colour |
|---|---|
| emergency | `#ef4444` (red) |
| urgent | `#f59e0b` (amber) |
| routine | `#10b981` (green) |

---

## Known Rough Edges to Watch

1. **Phase 9 router import path**: The router is in `src/loop3/`. Make sure
   `loop3` is importable from the FastAPI process. Check `pyproject.toml` —
   if it only lists `loop1` and `loop2` as packages, add `loop3`.

2. **Session not in memory**: If the server restarts, `_sessions` dict is
   cleared. `GET /api/session/{id}/routing` must fall back to loading the
   final record from disk (same as `_load_report_from_disk()`) if the session
   isn't live anymore.

3. **Emergency routing + booking**: Don't show a slot picker for emergency
   urgency. Show the ED alert instead. The booking endpoint should reject
   booking requests for emergency-level appointments.

4. **Doctor portal auth on page load**: `doctor-portal.html` must call
   `GET /api/me` on load and redirect to `login.html` if not authenticated
   or if role != doctor.

---

## Files to Create / Modify

| Action | File |
|---|---|
| CREATE | `web/data/doctors.json` |
| CREATE | `web/data/appointments.json` |
| CREATE | `web/static/doctor-portal.html` |
| MODIFY | `web/api.py` — add 5 new endpoints, doctor seed, appointments helpers |
| MODIFY | `web/static/report.html` — add Next Steps / booking section |
| MODIFY | `web/static/login.html` — role-based redirect after login |
| MODIFY | `web/static/auth.js` — store and expose role |
| MODIFY | `pyproject.toml` — add `loop3` to packages if missing |

---

## First-Session Checklist for Claude Code

1. Read this file fully.
2. Read `phase9_notes.md` to understand the routing engine.
3. Read `web/api.py` fully — understand the existing patterns before adding code.
4. Read `web/static/report.html` and `web/static/session.html` — understand the
   existing HTML/CSS patterns before adding new pages.
5. Read `web/static/auth.js` — understand how tokens and user state are stored.
6. Check `pyproject.toml` to confirm `loop3` is listed as a package.
7. Build Steps 1-10 in order. Test each step before moving on.
8. Do NOT modify any files in `src/loop1/` or `src/loop2/`.
9. Do NOT remove or rename existing API endpoints.
