
In a recommendation systems, there are 3 main components: candidate generator or retrieval, ranking, re-ranking

Some requirements;
1. it should be as fast as possible -> cache. And should be fast to repopulate that cache
2. it should be relevant as well

** Candidate generator
From the entire collection of items in the database, we want to generate a short list of items that user may want to check.
this process should be fast and efficient since the number of items is huge. And the result should be exhautive
There are number of ways to do that:
- Query the list of items in the channel that user already subscribed to (no ml-required)
- Query the list of items that are similar to what users already watched (easiest since it doesn't require info about the user)
- Query the list of items that similar user watched

The last two requires embedding generation for items and users. ML model + vector database can do that

** Ranking
From the above short list, now score assignment is now feasible. It's tempting to use the distance from candidate generation step to for scoring, but it's not recommended due to 2 reasons:
- There are normally several candidate generator which works in parallel to collect candidates using different criterias. so it doesn't make sense to compare the scores from different generators head-to-head
- Candidate generation is meant to be fast and efficient to generate an exhaustive list. To achieve that, it has to sacrifice the accuracy. Ranking model can be more sophisticated, consider more context to make the decision.
- Furthermore, candidate generation scoring focuses on the similarity between items. However, in this step, we may want to optimize other objective, like maximum click or maximize watch time

** Re-ranking
After making the second shortlist, we can apply some biz rules to produce the final recommendation. There are some criteria that one might consider:
1. Freshness
We can run the last 2 steps more often to have the latest items, or whenever a new item get uploaded, we compute the embebeding and then add to the cache instead of only running the process in batch. Or for the items in the subscribed channel, it will bypass the whole process and be present at the final list
2. Diversity
We can run the 2 above steps independently for different categories, then sample to get the final list