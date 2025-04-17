import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_file", type=str)
    parser.add_argument("--output_file", type=str)
    parser.add_argument("--test_file", type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    results = []
    error_line = 0
    for line_idx, line in enumerate(open(args.answers_file)):
        try:
            results.append(json.loads(line))
        except:
            error_line += 1

    results = {x['question_id']: x['text'][-1] for x in results}
    test_split = [json.loads(line) for line in open(args.test_file)]
    test_split_attributes = [x for x in test_split if x['category'] == 'direct_attributes']
    test_split_spatial = [x for x in test_split if x['category'] == 'relative_position']

    print(f'total results: {len(results)}, total split: {len(test_split)}, direct attributes: {len(test_split_attributes)}, relative position: {len(test_split_spatial)}, error_line: {error_line}')

    output = []
    count_spatial = 0
    count_attribute = 0
    for x in test_split:
        if x['question_id'] not in results:
            output.append({
                'question_id': x['question_id'],
                'category': x['category'],
                'label': x['label'],
                'answer': ''
            })
        else:
            if x['label'] == results[x['question_id']]:
                if x['category'] == 'direct_attributes':
                    count_attribute += 1
                elif x['category'] == 'relative_position':
                    count_spatial += 1
            else:
                output.append({
                    'question_id': x['question_id'],
                    'category': x['category'],
                    'label': x['label'],
                    'answer': results[x['question_id']]
                })

    with open(args.output_file, 'w') as f:
        json.dump(output, f)

    print(f'count_spatial: {count_spatial / len(test_split_spatial) * 100}, count_attribute: {count_attribute / len(test_split_attributes) * 100}, overall: {(count_spatial + count_attribute) / len(test_split) * 100}')