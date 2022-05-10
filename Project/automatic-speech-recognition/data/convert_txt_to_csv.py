import csv

def main():

    txt_file_in_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/mimic-recording-studio/backend/audio_files/1f1e4444-404b-ee95-bbb8-37eb2223425c/1f1e4444-404b-ee95-bbb8-37eb2223425c-metadata.txt'
    csv_file_out_path = '/home/nhinke/Documents/JHU/Robotics-MSE/S22/DL/Coursework/Project/automatic-speech-recognition/data/my-data/my-recorded-data.csv'

    csv_file_header = ['path', 'sentence', 'num characters']

    with open(txt_file_in_path, 'r') as txt_file:
        with open(csv_file_out_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter='\t')
            writer.writerow(csv_file_header)
            for line in txt_file:
                metadata = line.strip().split('|')
                writer.writerow(metadata)
            csv_file.close()
        txt_file.close()

if __name__ == "__main__":
    main()