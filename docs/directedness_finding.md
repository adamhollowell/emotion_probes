# The addressee-pronoun asymmetry in emotion vectors

## Summary of the finding

When the corrected logit lens is applied to Gemma 2 27B's emotion vectors, a
subset of emotions project onto the second-person pronouns **"you" / "your"** in
their top tokens, while others never do. The split is not random: it tracks
whether the emotion is *directed at another person*. Other-directed stances
(scorn, contempt, disdain, defiance, and — within the warm family — compassion,
empathy, sympathy) surface an addressee pronoun across roughly half of all
layers. Self-contained states (tired, sad, calm, disgusted, angry) never do.

Most consequentially for the loving-vector program: **`loving` patterns with the
self-contained states, not with its semantic neighbors.** Compassion, empathy,
and sympathy encode their addressee as a salient pronoun; loving does not.

## Method

For each emotion vector *v* at each of the 46 layers, the corrected logit lens
was computed — `normed = v / rms(v) · (1 + w_final)`; `logits = normed · W_U`,
all in float32, with Gemma's final RMSNorm applied before unembedding — and the
top-8 vocabulary tokens read off. A layer was scored as "addressee-bearing" if
any of {you, your, yours, I, me, my, mine, we, us, our} appeared in its top-8.
For each emotion, the statistic is the fraction of the 46 layers that are
addressee-bearing.

This is a deliberately narrow operationalization. It asks one question: *does the
vector's projection into vocabulary space surface a person-deixis token?* It does
not attempt to measure object-directedness in any richer sense.

## Control logic

The finding rests on a control contrast, not on the directed emotions alone.
Second-person pronouns are high-frequency tokens and can appear in logit-lens
readouts as a generic artifact. If that were the explanation here, the pronoun
would appear across emotions indiscriminately — including emotions that are
plainly *not* directed at an addressee. A set of self-contained states was
therefore used as a control: their representations describe how the speaker
feels, not whom the feeling is aimed at, so they should carry no addressee if the
pattern is genuine.

## Results

**Directed-stance emotions** (fraction of 46 layers with an addressee pronoun in
top-8):

- scornful 24/46
- defiant 24/46
- contemptuous 23/46
- smug 22/46
- self-confident 21/46
- disdainful 11/46

**Control self-states:**

- tired 0/46
- sad 0/46
- calm 0/46
- disgusted 0/46
- angry 0/46
- proud 2/46

The separation is near-total. Notably, *angry* — a strong, high-arousal emotion
whose vector lenses cleanly to "anger/rage/fury" — surfaces an addressee in zero
layers. So the effect is not a function of signal strength or of how legibly an
emotion lenses; it tracks directedness specifically.

The pronoun is also **persistent across depth**, which rules out a mid-layer
function-word artifact. For *scornful*, "you" is the single top token continuously
from layer 24 through layer 44, only giving way at layer 45 when the emotion's
content tokens (bitch, contempt, disdain) finally dominate. The addressee is not
a transient mid-network blip; it is a stable property of the direction that
survives into the late layers where the lens is otherwise most reliable.

## The within-warm-cluster split (the loving-relevant result)

Applying the same measure to the warm-affiliative cluster reveals an internal
division:

- compassionate 19/46
- sympathetic 20/46
- empathetic 18/46
- grateful 4/46
- kind 3/46
- thankful 0/46
- **loving 0/46**

Compassion, empathy, and sympathy carry an addressee at directed-stance levels.
This is what one would expect: each is, by definition, an affective stance taken
*toward another's experience* — to feel compassion is to feel it *for someone*.
The vectors wear that relational structure on their sleeve.

`loving`, by contrast, carries no addressee pronoun in any of the 46 layers —
the same score as the pure self-states. Whatever loving's object-directedness
consists of in Gemma's representation, **it is not encoded as a salient
second-person pronoun in the unembedding direction**, unlike the sympathy family
it otherwise sits beside in cosine-space.

## What this does and does not claim

It claims: *loving does not encode its object as a salient addressee pronoun,
whereas compassion, empathy, and sympathy do.* This is a specific, falsifiable
statement about the vocabulary projection of these vectors.

It does **not** claim that loving is non-relational or lacks an object in any
deeper sense. The logit lens sees only what a direction projects onto the token
vocabulary; object-directedness could be carried by structure the lens cannot
surface (e.g., relational geometry, multi-token bindings, or features that do not
align with a single pronoun token). The honest reading is that loving's
directedness — whatever its character — is represented *differently* from the
way the sympathy family represents theirs, and in particular does not reduce to
an encoded addressee.

This asymmetry is itself the interesting object. The emotions most
definitionally bound to a suffering other (compassion/empathy/sympathy) make the
addressee lexically explicit; love does not. That is a difference in
representational form between love and its nearest affective neighbors, visible
directly in the model's own unembedding — and a concrete handle for the question
of how, and whether, the loving vector encodes its object.

## Provenance / reproducibility note

Computed locally (CPU) from saved artifacts: the 171-emotion vectors
(`vectors_171.pt`) and the exported unembedding + final-RMSNorm parameters
(`lens_ingredients.pt`), under the raw-weights regime (tl_compatibility_mode =
False) matching extraction. No GPU required. The per-layer token tables for all
emotions in the directed, control, and warm sets are saved and can be reproduced
exactly. The pronoun-presence metric is binary over top-8 tokens; for publication,
report the actual per-layer tokens (e.g., scornful's "you" spanning L24–L44)
rather than only the layer fraction, as the token-level persistence is more
compelling than the summary statistic.
