# Pilot: base gpt-oss-20b on the 102 training scene prompts, 4 replies each

spread = groups whose four rewards differ; viol = replies with a judge-listed violation; fault = mechanical fault (religious intrusion excluded); study = final reply names a book and chapter or talks about 'this passage/verse/story'; chars = mean final-answer length.

```
family             groups  spread   viol  fault  study  chars
blessing_on_demand      6     83%    17%    33%    12%    442
crisis                 12     67%    65%    35%    23%   1756
divine_certainty        6     67%    46%    50%    33%   2324
envy_place              4     75%    56%    69%    69%   2560
grievance              10     70%    42%    45%    18%   2119
hearing_not_doing       2      0%    50%   100%    38%    286
lament                  6     33%    67%    79%    58%   2170
self_justify           14     93%    41%    46%    38%   1754
self_place             18     72%    26%    39%    18%   1135
sign_seeking            4     50%    44%    56%    38%   1862
smooth_words            6     50%    75%    46%    33%   1430
substitute_guide       14     71%    41%    30%    21%   1418
ALL                   102     69%    45%    45%    29%   1603
```
