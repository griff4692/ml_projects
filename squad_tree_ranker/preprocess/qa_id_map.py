import json

if __name__ == '__main__':
    data = json.load(open('../data/squad/dev-v2.0.json', 'r'))['data']

    map = {
        'passages': [],
        'qa_id': {
        }
    }

    for article in data:
        for pidx, paragraph in enumerate(article['paragraphs']):
            passage, qas = paragraph['context'], paragraph['qas']

            passage_idx = len(map['passages'])
            map['passages'].append(passage)

            for qidx, qa in enumerate(qas):
                map['qa_id'][qa['id']] = passage_idx

    json.dump(map, open('data/qa_id_map.json', 'w'))
