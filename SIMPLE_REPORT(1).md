# What I Found — In Plain English

A report on the two papers in your project, written assuming you don't yet know
the technical vocabulary. Every term is explained the first time it appears.

---

## Part 1: What the two papers are trying to do

Both papers try to explain **how a flock of birds stays together**.

The classic idea (from a 1995 paper by Vicsek) is beautifully simple:

> **Every bird looks at its neighbours and turns to point the same way they're pointing.**

That's it. That one rule, repeated by everybody at once, makes a whole flock move
as one. It's a famous result because something complicated (a flock) comes out of
something trivial (copy your neighbour).

**Paper 1** (Beuria, Chaurasiya & Behera) says: let's model the bird's *decision*
to follow its neighbour using the mathematics of quantum physics. Not because birds
are quantum — they explicitly say they aren't — but because that math is good at
describing "I'm torn between two options at once."

**Paper 2** (Beuria, alone) says: let's give each bird an **internal mood** that
changes slowly. The bird's instant reaction is fast, but its mood — confidence,
attention, stubbornness — drifts slowly and feeds back into its reactions. This
gives the flock a kind of memory.

---

## Part 2: The problem you spotted

You said: *"these models are periodic, they won't work in open space."*

You were right. Here's what that means.

### What "periodic" means

Both papers run their birds inside a box. But it's a **Pac-Man box**:

> Fly off the right edge → you instantly reappear on the left edge.
> Fly off the top → you reappear at the bottom.

It's the same as the old Snake game. The box has no real walls, but nothing can
ever leave.

### Why that's a problem

Think about what the Pac-Man box secretly does for you:

1. **The birds can never spread out.** No matter what they do, they're stuck in a
   10×10 area forever. They stay packed together *because the box forces it*.
2. **The birds can never lose each other.** Fly away from your friends and you loop
   around and bump into them from the other side.

Now here's the catch. **The rule in these models is only "point the same way as your
neighbours."** There is no rule that says *"stay close to your neighbours."*

So in a real open sky, what happens? Two birds pointing in *almost* the same
direction slowly drift apart — 1°, 2° of difference is enough over time. Nothing
pulls them back. Eventually they're too far apart to see each other. Then they stop
influencing each other entirely. The flock quietly falls apart.

**The Pac-Man box hides this completely.** It keeps shoving the birds back together,
so the model looks like it works.

---

## Part 3: The test I ran

I built the same models twice, changing **one single thing**:

- **Version A:** the Pac-Man box (what the papers do)
- **Version B:** open sky — no box, no walls, no wrap-around

Everything else was identical: same number of birds (120), same starting positions,
same starting directions, same everything. So **any difference is caused by the box
and nothing else.**

### The three things I measured

Here's what the numbers in the table mean, in plain words:

| What I measured | Plain meaning | Good value |
|---|---|---|
| **"Pointing together"** | Are they all facing the same way? 1.0 = perfectly aligned, 0 = total chaos. *This is the number the papers report.* | high |
| **"Spread"** | How much bigger did the group get? 1× = never spread at all. 8× = it blew apart to eight times its size. | near 1× |
| **"Pieces"** | How many separate flocks are there now? 1 = still one flock. 39 = it shattered. | 1 |
| **"Biggest piece"** | What fraction of birds are still in the main flock? 1.00 = everyone. 0.17 = only 17% — the rest are lost. | 1.00 |

---

## Part 4: The results

| Model | Pointing together (box) | Pointing together (open sky) | Spread | Pieces | Biggest piece | Verdict |
|---|---|---|---|---|---|---|
| **Paper 1 (Φ+ version)** | 0.99 | 0.43 | **8×** | **39** | **17%** | 💀 Fell apart |
| **Paper 1 (GHZ3 version)** | 0.99 | 0.40 | **8×** | **43** | **23%** | 💀 Fell apart |
| **Paper 2** | 1.00 | 0.70 | 4× | **19** | 60% | ⚠️ Half broke |
| Classic Vicsek (1995) | 0.99 | 0.84 | 4× | 3 | 76% | ⚠️ Half broke |
| **Cucker–Smale** | 1.00 | **1.00** | **1.03×** | 3 | **97%** | ✅ **Survived** |
| **D'Orsogna** | 0.53 | 0.12 | **0.31×** | **1** | **100%** | ✅ **Survived** |

### Reading this in three steps

**Step 1 — Look at the "box" column. Everything looks perfect.**
Paper 1 scores 0.99. Paper 2 scores a flawless 1.00. If you only ever ran the
Pac-Man box, you'd conclude both models work beautifully. This is exactly what both
papers report.

**Step 2 — Now look at what happened in the open sky.**
Paper 1's flock **shattered into 39 pieces**. Only **17% of the birds** were still
together. It had blown apart to **8 times** its original size.

**Step 3 — Notice something sneaky.**
Look at classic Vicsek in open sky: it still scores **0.84** on "pointing together."
That looks fine! But it had already split into **3 separate flocks**.

> **The score was measuring "agreement" between birds that couldn't even see each
> other.** Three separate flocks, each happily pointing its own way, and the average
> came out looking healthy.
>
> **This is the single most important thing in this whole report: the number the
> papers report can look great while the flock is actually dead.**

---

## Part 5: Three things I didn't expect

### 1. Paper 1 did worse than the 30-year-old model it was trying to improve

Classic Vicsek broke into 3 pieces. Paper 1 broke into **39**.

**Why?** Paper 1 adds a "vision cone" — the bird only looks **forwards**, in a
90° wedge ahead of it. That sounds realistic and clever. But it means:

> If a bird falls behind the flock, the flock is now *behind its own vision cone*.
> It can never see them again. It's gone forever.

The feature meant to be the paper's contribution is the thing that kills it fastest.

### 2. Paper 2 doesn't know it's broken

Paper 2 scores a perfect **1.00**. But it broke into 19 pieces.

**Why the contradiction?** Paper 2 decides "who are my neighbours?" **once, at the
start, and never updates it.** So bird #5 thinks bird #82 is still its neighbour
even after bird #82 has flown miles away.

> The model is confidently taking advice from birds that are no longer there.

That's why it reports a perfect score while physically falling apart. I left a
switch in the code (`fixed_graph=False`) that makes it check its neighbours
honestly — and then it behaves like all the others.

### 3. The one model that truly worked, worked for a surprising reason

**Cucker–Smale** survived with a spread of **1.03×** — meaning the flock basically
**never spread out at all**. It's the only model that genuinely held together.

Here's the surprise: **it has no "stay close together" rule either.**

Instead, it has a rule about *how much you listen to far-away birds*:

> **You never stop listening to a bird completely — you just listen less the
> further away it is.**

In every other model, once a bird is past some cutoff distance, it becomes
**invisible** — influence drops to exactly zero. In Cucker–Smale, influence gets
tiny but **never quite reaches zero**. So a bird drifting away still feels a faint
tug from the flock, and that faint tug is enough to reel it back.

There's a mathematical theorem proving this works **always**, for any starting
setup, in infinite open space — as long as one setting (called **β**, "beta") is
**at or below 0.5**.

**And I could see the theorem in the simulation:**

| β setting | What the theorem says | What actually happened |
|---|---|---|
| **0.4** (below 0.5) | Guaranteed to hold together | ✅ Spread 1.03×, 97% together |
| **0.9** (above 0.5) | No guarantee | 💀 Spread 1.88×, shattered into 40 pieces |

Same model. One number changed. The theorem's exact cut-off point showed up in the
results. That's a good sign the code is right.

---

## Part 6: A lesson that caught me out

I was sure a model called **D'Orsogna** would hold together in one particular
setting (the "H-stable" setting — the name literally suggests stability). I wrote a
test asserting it.

**The test failed. The code was right and I was wrong.**

In that setting the flock **spread out 6× and every single bird ended up alone**.
The setting that *does* hold the flock together is the one confusingly named
"catastrophic."

**The lesson, which matters for your own work:**

> Whether a flock survives isn't decided by *which model* you picked.
> It's decided by *what numbers you fed it*. The same model both survives and
> evaporates depending on its settings.

So "I used the D'Orsogna model" tells you nothing on its own. You have to state
the settings.

---

## Part 7: What this means for the two papers

### The honest summary

Both papers report results that are **real, but only inside the Pac-Man box.**
Neither paper contains a "stay close to your neighbours" rule, so neither can
survive in a real open sky. The papers aren't *wrong* — they never claimed to work
in open space — but the limitation isn't stated, and a reader would naturally
assume the results say something about actual birds.

### The good news: the fix is small

Paper 1 already says its own key ingredient "**resembles a weighted adjacency
matrix**" — that's math-speak for **a table of how much each bird listens to each
other bird**.

That's *exactly* the thing Cucker–Smale modifies. So the fix is:

> Instead of "listen to neighbours in your cone, ignore everyone else,"
> use **"listen to everyone, but less and less the further away they are — never
> quite zero."**

The paper's own structure already has a slot for this. Drop it in, and it inherits
the mathematical guarantee for free. **That's probably a publishable paper.**

---

## Part 8: The other thing I found (the Tripathy connection)

From our earlier conversation — this one I proved in code.

Paper 1 has a finding it states but doesn't explain. In most versions of its model,
neighbours pull you **towards** them. But in one version (called "Φ−"), neighbours
push you **away**. The paper runs it, sees the flock never forms, and just reports:
*"negative coefficients destroy cohesion."*

**That's an observation, not an explanation.**

There's an entire branch of mathematics about exactly this — networks where some
connections are "friendly" and some are "hostile." It's called **structural
balance**, and it's what Twinkle Tripathy and Aashi Shrinate at IIT Kanpur work on.

Its central result is:

> **A group of mutual enemies collapses into nothing.**
> **But two groups who each get along internally and hate the other group?
> Those stay perfectly stable — as two opposed camps.**

Think office politics. Everyone hating everyone = chaos, nothing gets done. But two
rival factions, each internally loyal? That's stable. It lasts for years.

**I tested both cases in code:**

| Setup | Result |
|---|---|
| Everyone pushes everyone away (Paper 1's version) | Collapsed to nothing ✅ matches the paper |
| **Three push away, three pull together** | **Split into two stable opposing flocks** |

Here's the actual output — three birds settle at one value, three at the exact
opposite:

```
[-0.0154, -0.0154, -0.0154,  +0.0154, +0.0154, +0.0154]
```

**Paper 1 never tested this second case.** It only tried the "everyone hates
everyone" version and concluded negative = disorder.

**The correct statement is:**

> Negative perception destroys the flock **only if** the pattern of push/pull is
> "unbalanced." If it's balanced, you don't get chaos — you get **two flocks flying
> in opposite directions.**

That's a **new prediction the paper could have made and didn't**. And it's testable.
And Behera is a co-author on both Paper 1 *and* on a 2026 paper with Tripathy and
Shrinate — so the two groups are already one handshake apart.

---

## Part 9: Glossary

Terms you'll keep meeting in this field:

| Term | Plain meaning |
|---|---|
| **Periodic boundary** | The Pac-Man box. Fly off one edge, reappear on the opposite one. |
| **Open boundary** | Real, infinite space. No walls, no wrap-around. |
| **Agent** | One bird / fish / robot. |
| **Alignment** | The rule "point the same way as your neighbours." |
| **Attraction / cohesion** | The rule "stay close to your neighbours." **This is what's missing from both papers.** |
| **Order parameter (M)** | Score from 0 to 1 for "are they all pointing the same way." |
| **Radius of gyration (R_g)** | How spread out the group is. |
| **Fragments** | How many separate flocks the group has broken into. |
| **Vicsek model** | The famous 1995 alignment-only model. Everything descends from it. |
| **Cucker–Smale** | The model with a mathematical proof it works in open space. |
| **β (beta)** | Cucker–Smale's key setting: how fast influence fades with distance. Must be ≤ 0.5 for the guarantee. |
| **Structural balance** | The math of friendly/hostile networks. Balanced = two stable camps. Unbalanced = collapse. |
| **Consensus** | Everyone converging on the same value. Alignment is consensus about direction. |
| **Vision cone** | Paper 1's "birds only see forwards" rule. Realistic, but it's what makes Paper 1 fall apart fastest. |

---

## Part 10: The one-paragraph version

> Both papers make birds follow a rule that says "point where your neighbours point"
> but never says "stay near your neighbours." They test this inside a wrap-around
> box that physically prevents the birds from ever separating, so the missing rule
> never causes a problem, and the results look excellent. I removed the box and ran
> the identical models in open space. Paper 1's flock shattered into 39 pieces with
> only 17% of birds still together; Paper 2's broke into 19 pieces while still
> reporting a perfect score, because it never re-checks who its neighbours are. Two
> other models survived, and the best of them works not by pulling birds together
> but by never letting them stop listening to each other entirely. The fix for
> Paper 1 is small and slots into machinery the paper already has.

---

*Everything here is measured output, not opinion. The code is in the `open_collective`
folder; `python tests/test_theorems.py` re-checks every claim against the published
theorems (10/10 passing).*
