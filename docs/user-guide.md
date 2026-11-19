# Playing LexForge

Run `lexforge --player Ada` to select a local profile. Choose menu numbers and press Enter. In submenus, type `0` to return. For text inputs, type `:back`. You can exit from the main menu, with Ctrl+C, or by closing input. Scored attempts are saved as soon as you complete them.

| Option | What you can do |
| --- | --- |
| 1 · Drafting challenge | Choose a difficulty, read the target and suggested details, and type your own continuation. The game joins it to the prompt and sends the completed text to GPT-2. After feedback, continue the set or return to the menu. |
| 2 · Next-word predictor | Enter a phrase and inspect up to five likely next words and their individual probabilities. Change bigram or trigram in Settings. |
| 3 · Analyze my legal sentence | Submit your own sentence and see GPT-2’s top three categories. Free drafting does not award points. |
| 4 · Guess the clause | Read an original example and identify its clause category. Each answer reveals the expected category. |
| 5 · Correct the clause | Rewrite a weak sample as a complete clause. The rubric lists the details that each exercise checks. |
| 6 · My progress | View XP, level progress, total attempts, distinct challenges, category averages, achievements, and the five latest attempts. |
| 7 · Leaderboard | Compare the top 20 local profiles by XP. |
| 8 · Model insights | Inspect held-out perplexities and classification metrics for the supplied artifacts. |
| 9 · Settings and help | Choose the next-word model order, change profiles, and see where your saves live. |
| 0 · Exit | Close the current session. |

## How scoring works

Drafts earn 20 points for a continuation of at least four words and 10 for sentence-ending punctuation. The two listed writing details are worth 20 each. A GPT-2 category match earns the final 30, regardless of classifier confidence. When GPT-2 is unavailable, the writing is scored out of the same 100-point scale and the final 30 points are marked unavailable. Replay later to attempt those points.

Revisions use the same 30 writing-form points and award 35 for each of the two exercise details. Guessing awards 100 for a correct category and zero otherwise. Writing points use visible criteria and recognizable expressions such as “written notice” and “notice in writing”. A valid alternative expression may miss a pattern, and keyword matches can pass without appropriate legal meaning.

XP equals the improvement over your previous best on a challenge. A first score of 70 earns 70 XP. A replay that scores 100 earns 30 more. Every 250 XP advances one level. Complete three distinct drafts to unlock Intermediate and six to unlock Advanced. Draft sets show unplayed items first. The guessing and revision sets prioritize examples without a perfect score.

The default SQLite file is `~/.lexforge/progress.sqlite3`. Use `--db PATH` to choose another file or `LEXFORGE_HOME` to choose its parent directory. Only gameplay statistics and player preferences are retained in SQLite. Draft text is used in the active session for scoring and inference.
