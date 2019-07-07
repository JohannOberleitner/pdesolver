import getopt
import json
import sys

def make_filename(baseFilename, gridSize, architectureType, count, epochs, index, postfix):
    return '{0}_{1}_{2}_{3}_{4}_{5}_{6}'.format(baseFilename, gridSize, architectureType, epochs, count, index, postfix)


def parseArguments(argv):
    supportedOptions = "hf:s:a:N:e:p:"
    usage = 'extract_metrics.py -f <basefileName> -s <gridSize> -a <architectureType> -N <count> -e <epochs> -p <postfix> --start <start> --end <end>'

    baseFileName = None
    label = None
    count = 20
    persistFile = None
    readFile = None

    try:
        opts, args = getopt.getopt(argv, supportedOptions, ["start=", "end="])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-f"):
            baseFileName = arg
        elif opt in ("-s"):
            gridSize = int(arg)
        elif opt in ("-a"):
            architectureType = int(arg)
        elif opt in ("-N"):
            count = int(arg)
        elif opt in ("-e"):
            epochs = int(arg)
        elif opt in ("-p"):
            postfix = arg
        elif opt in ("--start"):
            startIndex = int(arg)
        elif opt in ("--end"):
            endIndex = int(arg)

    return baseFileName, gridSize, architectureType, count, epochs, startIndex, endIndex, postfix


if __name__ == '__main__':

    baseFilename, gridSize, architectureType, count, epochs, startIndex, endIndex, postfix = parseArguments(sys.argv[1:])

    s = ''
    for index in range(startIndex, endIndex+1):
        filename = make_filename(baseFilename, gridSize, architectureType, count, epochs, index, postfix)

        with open(filename, "r") as read_file:
          data = json.load(read_file)

        avg_error_avg = data['avg_error_avg']
        avg_error_variance = data['avg_error_variance']
        variances_avg = data['variances_avg']
        variances_variance = data['variances_variance']
        median_values = data['median_values']
        median_values_variance = data['median_values_variance']
        max_values = data['max_values']
        max_values_variance = data['max_values_variance']

        learning_duration= data['learning_duration']

        #s = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(gridSize, count, architectureType, epochs, learning_duration, avg_error_avg,  avg_error_variance, variances_avg, variances_variance, median_values, median_values_variance, max_values, max_values_variance)
        #s1 = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t'.format(learning_duration, avg_error_avg,  avg_error_variance, variances_avg, variances_variance, median_values, median_values_variance, max_values, max_values_variance)
        #print(index, s1)

        s += '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t'.format(learning_duration, avg_error_avg,  avg_error_variance, variances_avg, variances_variance, median_values, median_values_variance, max_values, max_values_variance)

    print(s) # gridSize, count, architectureType, epochs, learning_duration, avg_error_avg,  avg_error_variance, variances_avg, variances_variance, median_values, median_values_variance, max_values, max_values_variance)
    #print(data)

