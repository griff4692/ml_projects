import json
import os
import argparse
import import_spinn

MINI_SIZE = 256
OVERWRITE = True

def gen_mini():
    data_dir = '.data/snli/snli_1.0'

    names = ['clean_train', 'clean_dev', 'clean_test']

    for name in names:
        source = 'snli_1.0_%s.jsonl' % name
        dest = 'snli_1.0_mini_%s.jsonl' % name

        source_file = os.path.join(data_dir, source)
        dest_file = os.path.join(data_dir, dest)

        if os.path.exists(dest_file) and not OVERWRITE:
            continue
        data = open(source_file, 'r').readlines()
        truncated_data = data[:MINI_SIZE]

        dest_fd = open(dest_file, 'w')
        for pt in truncated_data:
            dest_fd.write(pt)

def remove_unk(tag):
    source_file = '.data/snli/snli_1.0/snli_1.0_' + tag + '.jsonl'
    dest_file = '.data/snli/snli_1.0/snli_1.0_clean_' + tag + '.jsonl'

    if os.path.exists(dest_file) and not OVERWRITE:
        return

    data = open(source_file, 'r').readlines()

    orig_data_size = len(data)

    dest_fd = open(dest_file, 'w')
    removed = 0
    for pt in data:
        pt_json = json.loads(pt)
        label = pt_json['gold_label']
        if label == '-':
            removed += 1
            continue

        dest_fd.write(pt)

    print ("Removed Unk gold labels: shrunk dataset from %d to %d" % (orig_data_size, orig_data_size - removed))

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Preprocess arguments.')
    parser.add_argument('-gen_mini', action="store_true", default=False)
    args = parser.parse_args()
    if args.gen_mini:
        gen_mini()
    else:
        remove_train_unk()
