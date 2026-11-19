# Demo + conclusion script (Aditya)

Midpoint, October 30, 2026. About 7 minutes in total; timings are a guide.
Stage directions are in *italics*. Fill the `[brackets]` after your dry run.

---

## 0. Handover (5 s)

*Teammate finishes on the sentiment / technology slide.*

> Thanks, [teammate]. Now let me show you how all of this actually runs, from where the data lives to
> the dashboard an agent or QA lead would use.

---

## 1. Neon database: where everything lives (~1:00)

*Open the Neon console → project → Tables view. Have it open in a tab beforehand, already signed in.*
*Do **not** open rows of real calls. Stay on the table list, or open a synthetic table.*

> Everything we built sits in one shared Postgres database on Neon. That matters for two reasons:
> every teammate sees the same data, and every result is computed once and saved, so nothing gets
> recalculated or goes out of sync.
>
> It has two parts.
>
> *Point at the `big_synthetic` schema.*
> The first is `big_synthetic`, our practice CRM. It copies the structure of Salud's own tables
> (guarantors, accounts, contact history, receipts and quality reviews), filled with synthetic
> data: 2,000 guarantors, about 3,200 accounts and 11,000 contacts. We need this because Salud
> gave us the table layout but not the data, and it's what Phase 2's payment model will train on.
>
> *Point at the `phase1` schema.*
> The second is `phase1`, the call analysis. `calls` holds the 60 test calls: 10 real, redacted calls
> from Salud and 50 synthetic ones. For every call we store:
> - the AI review, in `ai_analyses`, `ai_checklist` and `ai_coaching`: the checklist answers with quotes, and the coaching tips;
> - the sentiment, in `turn_sentiment`, a score for every single turn, and `call_sentiment`, the call-level start, end and shift;
> - our team's review, in `reviewed_tags`, with the work split between us in `assignments`;
> - and `ai_analysis_versions`, which keeps every prompt version's output side by side so we can compare them.

*Optional, if you have a safe, prepared query:* show row counts, e.g. 60 calls, so the audience
sees it's real data without showing transcript text.

---

## 2. Manual tagging page: how we checked the AI (~1:15)

*Switch to the tagging site (agent-call-analysis-a5uabx4zm.vercel.app). Open a **synthetic** call.*
*Only show; **do not click save**. The tags are final.*

> Before we trust an AI with Salud's QA, we had to measure it. This is the tagging site we built for that.
>
> *Point at the checklist.*
> For each call, the AI first fills in every checklist item (Yes, No or Not applicable), each with
> a short reason and an exact quote from the call as evidence. A reviewer reads the transcript and
> either keeps the AI's answer or changes it. Every change is saved.
>
> The six of us reviewed all 60 calls, and every real call was reviewed by two different teammates,
> so we can also see how much *we* agree with each other.
>
> *Point at a quote.*
> The quotes keep the AI honest. If it claims the agent verified the patient's identity, it has to
> show us the line where that happened, and any quote that isn't actually in the transcript is
> thrown out automatically.
>
> One honest caveat: because reviewers started from the AI's answers, agreement here means "how
> many of the AI's answers our reviewers kept". That's why we also built the synthetic calls with
> the right answers written down *before* the AI saw them, which gives us an unbiased check.

---

## 3. What the tags told us, and how we refined the prompt (~1:00)

*Stay on the tagging site, or switch to slide 13 (0.94 agreement).*

> The written instructions we give the AI are called the prompt, and the tags let us score each version of it.
>
> With prompt version 2, reviewers kept **94%** of the AI's answers across 49 calls. Greeting and
> privacy compliance were perfect. The hardest item was the call outcome, around **83%**: things like
> whether the patient paid, promised to pay, or hung up.
>
> So we looked at every answer reviewers changed and wrote version 3, with stricter rules aimed at
> those mistakes. And it got *worse*, at **84%**. The new rules made the AI too lenient: it started
> turning correct "No" answers into "Yes".
>
> That was a useful lesson. Broad rules fix a few cases and break many more. So we kept version 2,
> and from here we only make small, targeted fixes, each one tested against the tags before it goes live.

---

## 4. The agent dashboard (~2:30)

*Switch to agent-call-analysis.vercel.app, already signed in. Zoom the browser to ~125%.*
*Pick a synthetic call you've rehearsed with.*

> This is the final product: the agent dashboard. It has three sections.

### Left: the calls (20 s)

*Point at the call list.*

> On the left are the transcripts, grouped into uploaded, real and synthetic calls, with a search box
> at the top. The colored dot shows how the customer *ended* the call (green positive, grey
> neutral, red negative), so a QA lead can spot the calls that went badly at a glance. A spinner means
> a call is still being analyzed.

### Middle: sentiment, summary, transcript (60 s)

*Click the synthetic call.*

> Opening a call loads its results straight from the database, with no waiting.
>
> *Point at the tiles across the top.*
> At the top are the sentiment scores: how the customer **ended** the call, the **shift** from start
> to end, the **QA score**, and the **call outcome**.
>
> *Point at the chart.*
> This chart is the caller's mood, turn by turn. These two buttons jump to the lowest and the highest
> moment in the call. *Click "Lowest".* Here the patient was [frustrated about X]…
> *click "Highest"* …and by here, they were [calmer / thanking the agent].
>
> That's exactly why we score the *end* of the call rather than one average: a patient who starts
> angry and leaves calm is a win for the agent, and a single average would hide it.
>
> *Point below the chart.*
> Below that is the AI summary, a few lines on what the call was about and how it ended. And below
> that, the full transcript, each turn shaded by its sentiment score.

### Right: QA checklist and coaching (60 s)

*Point at the right panel.*

> On the right is Salud's QA checklist: appropriate greeting, culture and values, captures key data,
> understands needs, correct information, first-contact resolution and privacy compliance. Each answer
> comes with a reason and a quote.
>
> *Click a checklist item / quote.* Clicking it highlights the exact turn in the transcript, so a
> reviewer can verify the AI in one click instead of rereading the whole call.
>
> The QA score is simply Yes divided by Yes plus No; Not-applicable items don't count against the agent.
>
> *Scroll to "Coaching for the agent".*
> Below the checklist are the coaching tips, concrete things the agent can do better next time. For
> this call: [read one tip].

### Adding a new call (30 s)

*Click **+ Upload calls**. Paste a short **made-up** transcript (prepare it beforehand) and click Upload.*

> Finally, new calls. You can upload transcript files or paste one in. It's saved to the database and
> analyzed automatically in the background. *Point at the spinner.* That's the spinner, and in a few
> seconds the sentiment, checklist and coaching appear, just like the call we just looked at.
> **Re-analyze** reruns both models on a call if we change the prompt.

*If the analysis is slow, don't wait. Say "it'll finish in the background" and move on.*

---

## 5. Conclusion (~1:00)

*Switch to slides: Next steps (14), Inputs from Salud (15), Conclusion (17).*

> To sum up Phase 1:
>
> 1. **Every call can now be reviewed**, not just a sample: automatically, on Salud's own checklist, in seconds.
> 2. **It's measurable and checkable.** Our reviewers kept 94% of the AI's answers, and every answer
>    is backed by a quote you can click and verify.
> 3. **It's safe by design.** Sentiment runs inside our app, so no call text leaves for it; the
>    checklist review uses an open-weight model, and we only send redacted transcripts.
>
> What's still open:
> - We score **7 of Salud's 9** checklist items. The other two, call status and grammar in notes,
>   need the agent's CRM disposition and notes, which the transcripts don't include.
> - With only 10 real calls, **more redacted calls from Salud**, ideally with those notes and the
>   payment that followed, would make these results much stronger.
>
> Next, in Phase 2, we join these call scores (quality, end-of-call sentiment and outcome) to
> each account's balance, age and payment history, and build the **payment propensity model**: a
> ranked list of which patients are most likely to pay, and who to call first. That's targeted for
> early December.
>
> Thank you. We're happy to take questions.

---

## Things you might have missed (add or prep these)

1. **Say "refine the prompt", not "fine-tune the model".** We never trained the model; fine-tuning
   means retraining, and a judge may pick on it.
2. **Frame 94% as a "kept rate", not accuracy**, and mention the synthetic calls with pre-written answers
   as the unbiased check (section 2 does this). Expect a question on it.
3. **The tiles row** (ended / shift / QA score / outcome) sits between the top bar and the chart. It's easy
   to skip when you say "sentiment on top".
4. **Lowest / Highest buttons and click-to-highlight quotes** are the most "wow" interactions. Make sure you actually click them.
5. **Upload a new call live.** It shows the system works on new calls, not just pre-loaded ones. Keep a made-up transcript in your clipboard.
6. **The 7-of-9 gap and the ask to Salud** belong in the conclusion; it's the clearest "input from client" ask.
7. **Link to Phase 2.** End by connecting call scores → payment propensity, so the demo isn't a dead end.

## Safety checklist before presenting

- [ ] Show **synthetic calls only** on camera. Never open a real Salud call, in Neon or on the dashboard.
- [ ] On the tagging site, **don't press save**.
- [ ] Upload only a **made-up** transcript (it's saved to the team DB and sent to Groq).
- [ ] Neon: don't show the connection string / "Connect" dialog on screen.
- [ ] All tabs open and signed in beforehand: Neon, tagging site, dashboard, slides. Browser at ~125%.
- [ ] Backup: a screen recording of the dashboard in case Wi-Fi or Vercel fails.

## Likely questions (short answers)

- **Why an LLM instead of ML?** Only 10 real calls; LLM with no training data scored kappa 0.66 vs 0.38 for the best classifier.
- **Isn't 94% inflated because reviewers saw the AI's answers?** Yes, it's a kept rate; that's why the synthetic calls have answers written beforehand. Against those, v2 and v3 tied at about 0.78.
- **Why did v3 get worse?** Broad leniency rules ("only say No if you can quote it") flipped correct No answers to Yes.
- **Does patient data leave?** Sentiment runs locally on CPU. The checklist review sends redacted text to GPT-OSS 120B (open-weight) on Groq; it could be self-hosted for production.
- **How is sentiment computed?** Per turn, P(positive) − P(negative), from −1 to +1; compare the first and last 30% of the patient's turns; ±0.25 for a clear label. End-of-call matched reviewers 75–80% vs about 60% for a whole-call average.
