
import csv
import re

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
            match = re.search("(\S+)\s+(\S+)", res)
            if not match:
                print(error)
                return
            label2result[label] = (match.group(1), match.group(2))
    return label2result

def ensemble(res1, res2, res3):
    cnt = {}
    score = {}

    cnt[res1[0]] = 0; cnt[res2[0]] = 0; cnt[res3[0]] = 0;
    score[res1[0]] = 0; score[res2[0]] = 0; score[res3[0]] = 0;

    cnt[res1[0]] += 1
    cnt[res2[0]] += 1
    cnt[res3[0]] += 1

    score[res1[0]] += float(res1[1])
    score[res2[0]] += float(res2[1])
    score[res3[0]] += float(res3[1])

    avgScore = (float(res1[1])+float(res2[1])+float(res3[1])) / 3

    best = 0
    score[best] = 0
    bestCnt = 0
    '''
    for c in cnt.keys():
        if cnt[c] > bestCnt or (cnt[c] == bestCnt and score[c] > score[best]):
            best = c
            bestCnt = cnt[c]
    '''
    for c in cnt.keys():
        if cnt[c] == 3:
            return c, 1 + avgScore
        elif cnt[c] == 2:
            return c, 0.6 + avgScore

    return res2[0], 0.3 + avgScore

def main():
    label2result1 = readResult('../experiment/landmarks-full/results.csv')
    label2result2 = readResult('../experiment/landmarks-full-256/rec_results.csv')
    label2result3 = readResult('../experiment/landmarks-full-densenet161/rec_results.csv')
    outputFile = open('./finalRes.csv', 'w')
    CSVwriter = csv.writer(outputFile)
    CSVwriter.writerow(('id', 'landmarks'))
    with open('../csvFiles/rec_test.csv', 'r') as csvFile:
        CSVreader = csv.reader(csvFile, skipinitialspace=True, delimiter=',')
        first = True
        for row in CSVreader:
            if first:
                first = False
                continue
            label = row[0]
            landmark, conf = ensemble(label2result1[label], label2result2[label], label2result3[label])
            CSVwriter.writerow((label, str(landmark) + ' ' + str(conf)))
    outputFile.close()


if __name__ == "__main__":
    main()

