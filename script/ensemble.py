
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

def ensemble(res1, res2, res3, res4, res5):
    cnt = {}
    score = {}

    cnt[res1[0]] = 0; cnt[res2[0]] = 0; cnt[res3[0]] = 0; cnt[res4[0]] = 0; cnt[res5[0]] = 0;
    score[res1[0]] = 0; score[res2[0]] = 0; score[res3[0]] = 0; score[res4[0]] = 0; score[res5[0]] = 0;

    cnt[res1[0]] += 1
    cnt[res2[0]] += 1
    cnt[res3[0]] += 1
    cnt[res4[0]] += 1
    cnt[res5[0]] += 1

    score[res1[0]] += float(res1[1])
    score[res2[0]] += float(res2[1])
    score[res3[0]] += float(res3[1])
    score[res4[0]] += float(res4[1])
    score[res5[0]] += float(res5[1])

    avgScore = (float(res1[1])+float(res2[1])+float(res3[1])+float(res4[1])+float(res5[1])) / 5

    '''
    best = 0
    score[best] = 0
    bestCnt = 0
    for c in cnt.keys():
        if cnt[c] > bestCnt or (cnt[c] == bestCnt and score[c] > score[best]):
            best = c
            bestCnt = cnt[c]
    '''
    max_c = ''
    max_vote = 0
    for c in cnt.keys():
        if cnt[c] > max_vote:
            max_vote = cnt[c]
            max_c = c

    if max_vote == 1:
        return res5[0], 0.2 + avgScore
    else:
        return c, max_vote / 5 + avgScore

def main():
    label2result1 = readResult('../experiment/landmarks-full/results.csv')
    label2result2 = readResult('../experiment/landmarks-full-256/rec_results.csv')
    label2result3 = readResult('../experiment/landmarks-full-densenet161/rec_results.csv')
    label2result4 = readResult('../experiment/landmarks-full-seresnet101/rec_results.csv')
    label2result5 = readResult('../experiment/landmarks-full-inceptionv3/rec_results.csv')
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
            landmark, conf = ensemble(label2result1[label], label2result2[label], label2result3[label], label2result4[label], label2result5[label])
            CSVwriter.writerow((label, str(landmark) + ' ' + str(conf)))
    outputFile.close()


if __name__ == "__main__":
    main()

