relevant_trigrams = [('IN', 'DT', 'NN'),
                     ('VB', 'JJ', 'NNS'),
                     ('VBZ', 'JJ', 'NNS'),
                     ('PRP', 'TO', 'VB'),
                     ('VB', 'DT', 'NN'),
                     ('DT', 'JJ', 'NNS'),
                     ('CC', 'JJ', 'NN'),
                     ('CC', 'PRP', 'VBZ'),
                     ('.', 'NN', 'VBP'),
                     ('TO', 'VB', 'IN'),
                     ('DT', 'NN', 'VBP'),
                     ('DT', 'NNS', 'VBP'),
                     ('PRP$', 'NN', 'CC'),
                     ('NN', '.', 'WRB'),
                     ('JJ', 'NN', 'CC'),
                     ('VBP', 'RB', 'JJ'),
                     ('TO', 'VB', 'JJR'),
                     ('VB', 'NN', 'IN'),
                     ('VBN', 'TO', 'VB'),
                     ('JJ', 'IN', 'PRP'),
                     ('NNS', '.', 'IN'),
                     ('PRP', 'VBP', 'JJ'),
                     ('IN', 'NN', '.'),
                     ('RB', ',', 'NN'),
                     (',', 'DT', 'NNS'),
                     ('NN', 'CC', 'TO'),
                     ('NNS', 'RB', 'VBP'),
                     ('JJ', 'NNS', ','),
                     ('NN', '.', 'IN'),
                     (',', 'IN', 'NNS'),
                     ('NN', 'IN', 'NNS'),
                     ('VBZ', 'DT', 'JJ'),
                     ('JJ', 'VBP', 'RB'),
                     ('VBP', 'DT', 'NN'),
                     (',', 'PRP', 'RB'),
                     ('JJ', 'NN', 'IN'),
                     ('NNS', 'VBP', 'JJ'),
                     ('VBZ', 'DT', 'NN'),
                     ('MD', 'VB', 'PRP'),
                     ('DT', 'NNS', '.'),
                     ('IN', 'PRP', 'VBZ'),
                     ('NN', 'TO', 'VB'),
                     ('VBZ', 'VBN', 'TO'),
                     ('NN', '.', 'NNS'),
                     ('PRP', 'MD', 'VB'),
                     ('PRP', 'VBD', 'DT'),
                     ('IN', 'PRP', 'TO'),
                     ('VB', 'IN', 'IN'),
                     (',', 'IN', 'PRP'),
                     ('RB', 'VB', 'NNS'),
                     ('VBP', 'RB', 'VB'),
                     ('RB', 'VB', 'NN'),
                     ('.', 'DT', 'NN'),
                     ('DT', 'NN', 'VBZ'),
                     ('NN', 'IN', 'DT'),
                     ('VBP', 'DT', 'JJ'),
                     ('VBG', 'JJ', 'TO'),
                     ('NNS', 'VBP', 'NN'),
                     ('NNS', ',', 'NN'),
                     ('NNS', 'IN', 'NN'),
                     ('NN', 'IN', 'NN'),
                     ('VBP', 'JJR', 'NN'),
                     ('VBD', 'TO', 'VB'),
                     ('VB', 'JJ', 'VBZ'),
                     ('JJR', 'NN', 'CC'),
                     ('NNS', '.', 'RB'),
                     ('NNS', 'WDT', 'VBP'),
                     ('VBG', 'PRP', 'TO'),
                     ('NN', ',', 'JJ'),
                     ('VBP', 'JJ', 'NN'),
                     ('NN', ',', 'CD'),
                     ('IN', 'PRP', 'RB'),
                     ('MD', 'VB', 'TO'),
                     (',', 'PRP', 'MD'),
                     ('IN', 'CD', 'NNS'),
                     (',', 'NN', 'VBP'),
                     ('DT', 'NN', 'IN'),
                     ('PRP', 'VBD', 'IN'),
                     ('JJ', 'NN', 'MD'),
                     ('NN', 'IN', 'PRP$'),
                     ('TO', 'NNS', 'MD'),
                     ('NN', '.', 'DT'),
                     ('NNS', 'JJ', 'IN'),
                     ('NNS', 'IN', 'DT'),
                     ('.', 'DT', 'JJ'),
                     ('PRP', 'NNS', ','),
                     ('NNS', ',', 'EX'),
                     ('IN', 'NN', ','),
                     ('NN', 'MD', 'VB'),
                     ('PRP', 'RB', '.'),
                     ('NNS', 'MD', 'VB'),
                     ('JJ', '.', 'RB'),
                     (',', 'PRP', 'VBD'),
                     ('NNS', 'TO', 'VB'),
                     ('NN', 'VBZ', 'PRP'),
                     ('NNS', 'IN', 'PRP'),
                     ('VBD', 'DT', 'JJ'),
                     ('WP', 'MD', 'VB'),
                     ('IN', 'VBG', 'CC'),
                     ('IN', 'NN', 'IN'),
                     ('JJ', ',', 'VBG'),
                     ('MD', 'VB', 'NNS'),
                     ('CC', 'WRB', 'PRP'),
                     ('DT', 'NNS', 'IN'),
                     ('WRB', 'PRP', 'VBP'),
                     ('DT', 'NNS', 'VBD'),
                     ('RB', 'VB', 'IN'),
                     ('NN', 'DT', 'NN'),
                     ('DT', 'NN', '.'),
                     ('CC', 'VBG', 'IN'),
                     ('VBP', 'JJR', 'NNS'),
                     ('.', 'IN', 'IN'),
                     ('IN', 'PRP$', 'NN'),
                     ('VB', 'PRP$', 'NN'),
                     ('.', 'DT', 'MD'),
                     ('RB', ',', 'PRP'),
                     ('IN', 'DT', 'JJ'),
                     ('.', 'IN', 'NN'),
                     (',', 'PRP', 'VBP')]

relevant_trigram_set = set(relevant_trigrams)
transition_words = [('and', 'then'),
                    ('besides'),
                    ('equally', 'important'),
                    ('finally'),
                    ('further'),
                    ('furthermore'),
                    ('nor'),
                    ('next'),
                    ('lastly'),
                    ('what\'s', 'more'),
                    ('moreover'),
                    ('in', 'addition'),
                    ('first'),
                    ('second'),
                    ('third'),
                    ('fourth'),
                    ('whereas'),
                    ('yet'),
                    ('on', 'the', 'other', 'hand'),
                    ('however'),
                    ('nevertheless'),
                    ('on', 'the', 'contrary'),
                    ('by', 'comparison'),
                    ('compared', 'to'),
                    ('up', 'against'),
                    ('balanced', 'against'),
                    ('vis', 'a', 'vis'),
                    ('although'),
                    ('conversely'),
                    ('meanwhile'),
                    ('after', 'all'),
                    ('in', 'contrast'),
                    ('although', 'this', 'may', 'be', 'true'),
                    ('because'),
                    ('since'),
                    ('for', 'the', 'same', 'reason'),
                    ('obviously'),
                    ('evidently'),
                    ('indeed'),
                    ('in', 'fact'),
                    ('in', 'any', 'case'),
                    ('that', 'is'),
                    ('still'),
                    ('in', 'spite', 'of'),
                    ('despite'),
                    ('of', 'course'),
                    ('once', 'in', 'a', 'while'),
                    ('sometimes'),
                    ('immediately'),
                    ('thereafter'),
                    ('soon'),
                    ('after', 'a', 'few', 'hours'),
                    ('then'),
                    ('later'),
                    ('previously'),
                    ('formerly'),
                    ('in', 'brief'),
                    ('as', 'I', 'have', 'said'),
                    ('as', 'I', 'have', 'noted'),
                    ('as', 'has', 'been', 'noted'),
                    ('definitely'),
                    ('extremely'),
                    ('obviously'),
                    ('absolutely'),
                    ('positively'),
                    ('naturally'),
                    ('surprisingly'),
                    ('always'),
                    ('forever'),
                    ('perennially'),
                    ('eternally'),
                    ('never'),
                    ('emphatically'),
                    ('unquestionably'),
                    ('without', 'a', 'doubt'),
                    ('certainly'),
                    ('undeniably'),
                    ('without', 'reservation'),
                    ('following', 'this'),
                    ('at', 'this', 'time'),
                    ('now'),
                    ('at', 'this', 'point'),
                    ('afterward'),
                    ('subsequently'),
                    ('consequently'),
                    ('previously'),
                    ('before', 'this'),
                    ('simultaneously'),
                    ('concurrently'),
                    ('thus'),
                    ('therefore'),
                    ('hence'),
                    ('for', 'example'),
                    ('for', 'instance'),
                    ('in', 'this', 'case'),
                    ('in', 'another', 'case'),
                    ('on', 'this', 'occasion'),
                    ('in', 'this', 'situation'),
                    ('take', 'the', 'case', 'of'),
                    ('to', 'demonstrate'),
                    ('to', 'illustrate'),
                    ('as', 'an', 'illustration'),
                    ('on', 'the', 'whole'),
                    ('summing', 'up'),
                    ('to', 'conclude'),
                    ('in', 'conclusion'),
                    ('as', 'I', 'have', 'shown'),
                    ('as', 'I', 'have', 'said'),
                    ('accordingly'),
                    ('as', 'a', 'result')]

transitions_set = set(transition_words)