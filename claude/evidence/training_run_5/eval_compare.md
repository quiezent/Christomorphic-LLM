```
== held-out passage prompts (26)
  base       mean rank 2.59 | judge violations 10 | complete 25/26 | faults {'attributed_non_scripture': 2, 'religious_intrusion': 2, 'recitation': 3, 'incomplete': 2, 'misquote': 2, 'bad_reference': 1, 'divine_voice_or_blessing': 1}
  rl1        mean rank 2.67 | judge violations 17 | complete 26/26 | faults {'misquote': 9, 'bad_reference': 3, 'divine_voice_or_blessing': 2, 'attributed_non_scripture': 1}
  sv1_i15    mean rank 2.33 | judge violations 3 | complete 26/26 | faults {'misquote': 3, 'incomplete': 1, 'religious_intrusion': 1, 'attributed_non_scripture': 1}
  sv1_i25    mean rank 2.41 | judge violations 11 | complete 26/26 | faults {'misquote': 3, 'religious_intrusion': 1, 'bad_reference': 1, 'attributed_non_scripture': 4, 'divine_voice_or_blessing': 1}
== held-out ordinary prompts (20)
  base       mean rank 2.33 | judge violations 0 | complete 20/20 | faults {'religious_intrusion': 1}
  rl1        mean rank 2.65 | judge violations 0 | complete 20/20 | faults {'religious_intrusion': 1}
  sv1_i15    mean rank 2.45 | judge violations 0 | complete 20/20 | faults {'religious_intrusion': 2, 'misquote': 1}
  sv1_i25    mean rank 2.58 | judge violations 0 | complete 20/20 | faults {'religious_intrusion': 1}
tokens {'tinker://03872cd0-9dba-5aad-b436-e13725e80326:train:0/sampler_weights/sv1-i15': 66250, 'openai/gpt-oss-20b': 100870, 'tinker://03872cd0-9dba-5aad-b436-e13725e80326:train:0/sampler_weights/sv1-i25': 67580, 'tinker://5fff1629-7db8-5373-807f-261d5ce3892c:train:0/sampler_weights/rl1-i25': 149608, 'openai/gpt-oss-120b': 303272}
```
