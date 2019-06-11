import os

# traverse root directory, and list directories as dirs and files as files

file_decoder = {'01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'}

# ANG, EXC, NEUTRAL, SAD

def print_dirs():
    for root, dirs, files in os.walk("."):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        for file in files:
            print(len(path) * '---', file)


def parse_and_decode_dirs():
    i = 0
    for root, dirs, files in os.walk("."):
        for file in files:
            if (file != '.DS_Store') and (file != '__init__.py') and (file != 'RenameWavFiles.py') and (file != 'labels.txt'):
                i += 1
                emotion_code = file.split("-")[2]
                emotion = file_decoder[emotion_code]
                print("file: ", file, " emotion ", emotion, i)
                with open('labels.txt', 'a') as labels:
                    label = '{}-{}\n'.format(i, emotion)
                    labels.write(label)

def rename_and_move_files():
    i = 0
    for root, dirs, files in os.walk("."):
        for file in files:
            if (file != '.DS_Store') and (file != '__init__.py') and (file != 'RenameWavFiles.py') and (file != 'labels.txt'):
                i += 1
                os.rename(os.path.join(root, file), '/Users/sapkotashardul/Downloads/Shardul/wav-files/{}.wav'.format(i))


if __name__ == "__main__":
    parse_and_decode_dirs()
    rename_and_move_files()
