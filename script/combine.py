
import csv

def readResult(fileName):
    label2result = {}
    with open(fileName, 'r') as csvFile:
        CSVreader = csv.reader(csvFile, skipinitialspace=True, delimiter=',')
        first = True
        for row in CSVreader:
            if first:
                first = False
                continue
            label = row[0]
            res = row[1]
            label2result[label] = res
    return label2result

def main():
    label2result = readResult('./results.csv')
    outputFile = open('./finalRes.csv', 'w')
    CSVwriter = csv.writer(outputFile)
    with open('./test.csv', 'r') as csvFile:
        CSVreader = csv.reader(csvFile, skipinitialspace=True, delimiter=',')
        first = True
        for row in CSVreader:
            if first:
                first = False
                continue
            label = row[0]
            if label in label2result.keys():
                CSVwriter.writerow((label, label2result[label]))
            else:
                CSVwriter.writerow((label, '0 0.0'))
    outputFile.close()


if __name__ == "__main__":
    main()

