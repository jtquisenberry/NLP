entities = \
[['T1', 'Location', [[0, 10]]],
 ['T2', 'Other', [[10, 11]]],
 ['T3', 'Location', [[12, 22]]],
 ['T4', 'Other', [[22, 23]]],
 ['T5', 'Other', [[24, 27]]],
 ['T6', 'Location', [[28, 31]]],
 ['T7', 'Location', [[32, 38]]],
 ['T8', 'Other', [[39, 42]]],
 ['T9', 'Other', [[43, 49]]],
 ['T10', 'Other', [[50, 52]]],
 ['T11', 'Other', [[53, 56]]],
 ['T12', 'Location', [[57, 58]]],
 ['T13', 'Other', [[58, 59]]],
 ['T14', 'Location', [[59, 60]]],
 ['T15', 'Other', [[60, 61]]],
 ['T16', 'Location', [[62, 68]]],
 ['T17', 'Location', [[69, 73]]],
 ['T18', 'Other', [[74, 76]]],
 ['T19', 'Other', [[77, 80]]],
 ['T20', 'Other', [[80, 81]]]]

sequence = 'California, Washington, and New Mexico are states in the U.S. Puerto Rico is not.'

def combine_entities(entities, sequence):

    combined_entities = []
    previous_type = 'Other'
    previous_token = ''
    keep_adding = False

    for entity_number, entity in enumerate(entities):
        # Append or merge
        # append
        if entity_number == 11:
            breakpointA = 0
        token = sequence[entity[2][0][0]:entity[2][0][1]]
        token_type = entity[1]
        if not keep_adding:
            combined_entities.append(entity)
            if token_type == 'Location':
                keep_adding = True
            else:
                keep_adding = False
        else:
            if token_type == 'Location':
                if previous_type == 'Location':
                    combined_entities[-1][2][0][1] = entity[2][0][1]
                    keep_adding = True
                elif previous_type == 'Other' and len(token) == 1:
                    combined_entities[-1][2][0][1] = entity[2][0][1]
                    keep_adding = True
                else:
                    combined_entities.append(entity)
            elif token_type == 'Other' and token == '.' and len(previous_token) == 1 and previous_type == 'Location':
                combined_entities[-1][2][0][1] = entity[2][0][1]
                keep_adding = True
            else:
                combined_entities.append(entity)
                keep_adding = False

        previous_token = token
        previous_type = token_type

        aaa = 1
    bbb = 1
    for e in combined_entities:
        print(e)







if __name__ == '__main__':
    combine_entities(entities, sequence)