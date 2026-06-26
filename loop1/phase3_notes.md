Phase 3 priorities (empirical observations from Phase 2):

1. Profile cleanup during compression
   Problem observed: After 10-15 turns, symptoms list grows to 8-12 entries
   with significant duplication. E.g., "chest pain" and "chest pain at rest"
   as separate entries; "rash" notes field contained the same phrase 3 times.
   
   Required behavior in compressor:
   - Before generating running_summary, deduplicate symptoms list:
     - Merge entries whose names match case-insensitively
     - Merge entries where one name is a substring of another (e.g.,
       "headache" and "headache triggered by tedious work" → keep "headache",
       move qualifier into notes)
     - Deduplicate substring repetitions within a single notes field
   - Same dedup logic for history.* lists
   - free_notes: strip duplicate sentences; preserve only unique observations

2. History categorization is unreliable
   Problem observed: profile_updater puts items in wrong categories.
   Examples: "weight loss" in social history; "blood clots" in social history
   instead of medical; "overdue MMR" in symptoms instead of medical.
   
   Required behavior: Either the compressor performs a re-categorization pass,
   OR add an explicit categorization step before compression. The compressor
   has the global view (full turn history) which the profile_updater lacks
   (single-turn view), so compressor is the right place to fix this.

3. Token monitoring
   Need data: log prompt token count per turn. After Phase 3, run one of the
   existing Case 1 logs through both the old and new pipeline and compare.

4. Compressor must not invent information
   The compressor is itself an LLM call. Prompt it to extract and reorganize
   only; explicit instruction to never add facts not present in source turns.