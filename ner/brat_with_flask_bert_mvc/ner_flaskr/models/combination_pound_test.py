entities = \
[['T1', 'Person', [[0, 3]]],
 ['T2', 'Person', [[3, 5]]],
 ['T3', 'Other', [[6, 11]]],
 ['T4', 'Other', [[12, 14]]],
 ['T5', 'Other', [[15, 18]]],
 ['T6', 'Location', [[19, 26]]],
 ['T7', 'Location', [[27, 29]]],
 ['T8', 'Location', [[30, 33]]],
 ['T9', 'Location', [[34, 40]]],
 ['T10', 'Location', [[41, 44]]],
 ['T11', 'Location', [[44, 45]]],
 ['T12', 'Other', [[45, 46]]]]

sequence = 'Mayle lives in the Kingdom of the Yellow Butt.'

tokens = \
['May',
 '##le',
 'lives',
 'in',
 'the',
 'Kingdom',
 'of',
 'the',
 'Yellow',
 'But',
 '##t',
 '.']




def combine_pound_entities(entities, tokens, sequence):

    previous_token = ''
    previous_type = 'Other'
    previous_end = -1
    combined_entities = []
    for entity_number, entity in enumerate(entities):

        token = tokens[entity_number]

        # Adjacent tokens
        current_start = entity[2][0][0]
        if current_start == previous_end:
            # Either the previous or the current has ##
            if (len(previous_token) >= 3 and previous_token[-2:] == '##') or (len(token) >= 3 and token[:2] == '##'):
                # Merge
                combined_entities[-1][2][0][1] = entity[2][0][1]
            else:
                combined_entities.append(entity)
        else:
            combined_entities.append(entity)

        previous_end = entity[2][0][1]

    for x in combined_entities:
        print(x, sequence[x[2][0][0]:x[2][0][1]])




if __name__ == '__main__':
    combine_pound_entities(entities=entities, tokens=tokens, sequence=sequence)





