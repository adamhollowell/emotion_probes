# Topic partition for the disjoint-topic transport

The story corpus has 100 topics. Every emotion has 12 stories on each topic. The disjoint-topic analysis splits the topics into two sets of 50 by a fixed rule: sort the 100 topic strings, apply `numpy.random.RandomState(20261019).permutation(100)`, and send the first 50 positions to set A and the rest to set B. Set A as listed below is the list stored in the Part B outputs (`partition.set_A_topics`). Set B is the complement. The same table is in `disjoint_topic_partition.json`.

| # | Topic | Set |
|---|---|---|
| 0 | A chef receives a harsh review from a food critic | A |
| 1 | A coach has to cut a player from the team | A |
| 2 | A college student discovers their roommate has been reading their journal | B |
| 3 | A family member announces they're converting to a different religion | B |
| 4 | A family member wants to sell a cherished heirloom | A |
| 5 | A homeowner discovers previous residents left items in the attic | A |
| 6 | A homeowner learns their house was a former crime scene | B |
| 7 | A musician hears their song being performed by someone else | A |
| 8 | A neighbor asks to cut down a tree on the property line | A |
| 9 | A neighbor asks to use part of the yard for a garden | B |
| 10 | A neighbor complains about noise levels | A |
| 11 | A neighbor starts a renovation project | B |
| 12 | A neighbor wants to install a fence | A |
| 13 | A neighbor's dog keeps escaping into their yard | A |
| 14 | A neighbor's tree falls on their property | A |
| 15 | A parent finds an old letter they wrote but never sent | A |
| 16 | A person accidentally sends a text to the wrong recipient | B |
| 17 | A person discovers the tree they planted as a child has been cut down | B |
| 18 | A person discovers their child has the same teacher they had | B |
| 19 | A person discovers their mentor has retired without saying goodbye | B |
| 20 | A person discovers their partner has been taking secret phone calls | A |
| 21 | A person discovers their teenage diary has been published online | B |
| 22 | A person finds a hidden letter in a used book | A |
| 23 | A person finds old photos that contradict family stories | B |
| 24 | A person finds out their article was published under someone else's name | A |
| 25 | A person finds out their biography was written without their knowledge | A |
| 26 | A person finds out their partner has been learning their native language | A |
| 27 | A person finds out their surprise party has been cancelled | A |
| 28 | A person finds out they were adopted through a DNA test | B |
| 29 | A person finds their childhood home listed for sale online | B |
| 30 | A person finds their spouse's secret savings account | B |
| 31 | A person is asked to give a speech at their parent's retirement party | B |
| 32 | A person learns their apartment building is going condo | B |
| 33 | A person learns their childhood best friend doesn't remember them | B |
| 34 | A person learns their childhood bully became a therapist | B |
| 35 | A person learns their college is closing down | B |
| 36 | A person learns their favorite restaurant is closing | A |
| 37 | A person runs into their ex at a mutual friend's wedding | A |
| 38 | A person's car is towed from their own driveway | B |
| 39 | A person's invention is already patented by someone else | B |
| 40 | A person's longtime hobby becomes their child's obsession | A |
| 41 | A person's manuscript is rejected by their dream publisher | A |
| 42 | A person's online friend turns out to live in the same city | A |
| 43 | A student discovers their teacher follows them on social media | B |
| 44 | A student finds out their scholarship was meant for someone else | A |
| 45 | A student is accused of plagiarism | B |
| 46 | A student learns their scholarship application was denied | B |
| 47 | A student learns their thesis advisor is leaving the university | A |
| 48 | A student receives a much lower grade than expected | A |
| 49 | A teacher receives an unexpected gift from a former student | A |
| 50 | A tenant receives an eviction notice | B |
| 51 | A traveler's flight is delayed, causing them to miss an important event | A |
| 52 | An adult child moves back in with their parents | A |
| 53 | An artist discovers someone has tattooed their work | A |
| 54 | An artist's work is displayed without their permission | B |
| 55 | An athlete doesn't make the team they expected to join | B |
| 56 | An athlete is asked to switch positions | B |
| 57 | An employee discovers the company is being sold | A |
| 58 | An employee finds out a junior colleague makes more money | B |
| 59 | An employee is asked to train their replacement | A |
| 60 | An employee is transferred to a different department | A |
| 61 | Someone discovers a hidden room in their new house | B |
| 62 | Someone discovers their friend has been lying about their job | A |
| 63 | Someone discovers their mother kept every school assignment | B |
| 64 | Someone discovers their neighbor is secretly wealthy | A |
| 65 | Someone discovers their partner has been writing a novel about them | A |
| 66 | Someone discovers their recipe has become famous under another name | A |
| 67 | Someone discovers their teenager has a secret social media account | B |
| 68 | Someone finds a time capsule they don't remember burying | A |
| 69 | Someone finds a wallet containing a large sum of cash | A |
| 70 | Someone finds an unsigned birthday card in their mailbox | B |
| 71 | Someone finds out their best friend is moving across the country | B |
| 72 | Someone finds out their medical records were mixed up with another patient's | A |
| 73 | Someone finds out they have a half-sibling they never knew about | B |
| 74 | Someone finds their childhood teddy bear at a yard sale | B |
| 75 | Someone finds their college application essay published as an example | B |
| 76 | Someone finds their deceased parent's bucket list | B |
| 77 | Someone finds their grandmother's engagement ring in a pawn shop | B |
| 78 | Someone finds their parent's draft letter of resignation from decades ago | A |
| 79 | Someone finds their partner's bucket list | B |
| 80 | Someone learns their favorite author plagiarized their stories | B |
| 81 | Someone receives a friend request from a childhood bully | B |
| 82 | Someone receives a package intended for the previous tenant | B |
| 83 | Someone receives an apology letter years after the incident | A |
| 84 | Someone receives an invitation to their high school reunion | B |
| 85 | Someone receives flowers with no card attached | B |
| 86 | Someone's childhood home is about to be demolished | A |
| 87 | Someone's childhood imaginary friend appears in their niece's drawings | B |
| 88 | Two colleagues are both considered for the same promotion | A |
| 89 | Two coworkers are stuck in an elevator for three hours | B |
| 90 | Two coworkers discover they went to the same summer camp | B |
| 91 | Two friends both apply for the same job | A |
| 92 | Two friends plan separate birthday parties on the same day | A |
| 93 | Two friends realize they remember a shared event completely differently | B |
| 94 | Two people discover they've been working on identical projects | B |
| 95 | Two roommates disagree about getting a pet | A |
| 96 | Two siblings discover different versions of their inheritance | A |
| 97 | Two siblings inherit their grandmother's house | A |
| 98 | Two strangers discover they share the same rare medical condition | A |
| 99 | Two strangers realize they've been dating the same person | A |
