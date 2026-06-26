Residual Phase 3 issues (accepted, address later):

1. Symptoms appearing in history.medical
   - Compressor putting current presenting symptoms in history.medical
   - Example: "chest pain", "shortness of breath" appearing in history.medical
   - Fix in Phase 6 or via stronger compressor prompt rules

2. Symptom variant over-listing
   - "leg swelling" + "left leg pain" + "knee pain" as three entries when 
     they're three descriptions of one symptom
   - Phase 4 ICL may improve via consistent symptom naming
   - Otherwise address in Phase 6 with more aggressive merging

3. Notes-field append duplication
   - "at rest" appearing twice in chest pain notes
   - Sentence-level dedup within notes field needs improvement
   - Minor; Phase 6 if it persists