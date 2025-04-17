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

    results = {x['question_id']: x['text'] for x in results}
    test_split = [json.loads(line) for line in open(args.test_file)]
    test_split_attribute = [x for x in test_split if x['category'] == 'attribute']
    test_split_ocr = [x for x in test_split if x['category'] == 'OCR']
    test_split_existence = [x for x in test_split if x['category'] == 'existence']
    test_split_count = [x for x in test_split if x['category'] == 'count']

    print(f'total results: {len(results)}, total split: {len(test_split)}, attribute: {len(test_split_attribute)}, OCR: {len(test_split_ocr)}, existence: {len(test_split_existence)}, count: {len(test_split_count)}, error_line: {error_line}')

    output = []
    count_ocr = 0
    count_attribute = 0
    count_existence = 0
    count_count = 0
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
                if x['category'] == 'attribute':
                    count_attribute += 1
                elif x['category'] == 'OCR':
                    count_ocr += 1
                elif x['category'] == 'existence':
                    count_existence += 1
                elif x['category'] == 'count':
                    count_count += 1
            else:
                output.append({
                    'question_id': x['question_id'],
                    'category': x['category'],
                    'label': x['label'],
                    'answer': results[x['question_id']]
                })

    with open(args.output_file, 'w') as f:
        json.dump(output, f)

    print(f'count_attribute: {count_attribute / len(test_split_attribute) * 100}, count_ocr: {count_ocr / len(test_split_ocr) * 100}, count_existence: {count_existence / len(test_split_existence) * 100}, count_count: {count_count / len(test_split_count) * 100}, overall: {(count_attribute + count_ocr + count_existence + count_count) / len(test_split) * 100}')