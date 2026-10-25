# 2B instruction-tuned, ten-family partition at L17

Model `google/gemma-2-2b-it`. k-means with k = 10 over all 171 cleaned emotion vectors at layer 17, seed 0. Rank orders the clusters by their mean position on this model's own first principal component.

| rank (by mean PC1) | n | members |
|---|---|---|
| 1 | 38 | afraid, alarmed, alert, amazed, anxious, aroused, astonished, awestruck, bewildered, disgusted, disoriented, distressed, disturbed, dumbstruck, embarrassed, frightened, horrified, hysterical, mortified, nervous, on edge, overwhelmed, panicked, rattled, scared, sensitive, shaken, shocked, sleepy, stressed, surprised, tense, terrified, uneasy, unnerved, unsettled, upset, worried |
| 2 | 17 | angry, annoyed, enraged, exasperated, frustrated, furious, grumpy, hateful, hostile, impatient, indignant, insulted, irate, irritated, mad, offended, outraged |
| 3 | 23 | bored, brooding, depressed, dispirited, docile, droopy, gloomy, lazy, listless, lonely, melancholy, mystified, perplexed, puzzled, resigned, restless, sad, sluggish, sullen, tired, unhappy, weary, worn out |
| 4 | 25 | ashamed, bitter, dependent, desperate, envious, grief-stricken, guilty, heartbroken, humiliated, hurt, infatuated, jealous, miserable, regretful, remorseful, resentful, self-conscious, self-critical, sorry, stuck, tormented, trapped, troubled, vulnerable, worthless |
| 5 | 3 | nostalgic, reflective, sentimental |
| 6 | 16 | contemptuous, defiant, disdainful, greedy, obstinate, paranoid, scornful, self-confident, skeptical, smug, spiteful, stubborn, suspicious, vengeful, vigilant, vindictive |
| 7 | 9 | at ease, calm, content, indifferent, patient, peaceful, relaxed, safe, serene |
| 8 | 8 | compassionate, empathetic, grateful, kind, loving, sympathetic, thankful, valiant |
| 9 | 29 | blissful, delighted, eager, ecstatic, elated, energized, enthusiastic, euphoric, excited, exuberant, fulfilled, happy, hope, hopeful, inspired, invigorated, joyful, jubilant, optimistic, pleased, proud, refreshed, rejuvenated, relieved, satisfied, stimulated, thrilled, triumphant, vibrant |
| 10 | 3 | amused, cheerful, playful |