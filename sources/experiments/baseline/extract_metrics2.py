import getopt
import json
import sys

def make_filename(baseFilename, gridSize, architectureType, count, epochs, index, charges_count, postfix):
    return '{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}'.format(baseFilename, gridSize, architectureType, charges_count, epochs, count, index, postfix)


def parseArguments(argv):
    supportedOptions = "hf:s:a:N:e:p:c:"
    usage = 'extract_metrics.py -f <basefileName> -s <gridSize> -a <architectureType> -N <count> -e <epochs> -c <charges> -p <postfix> --start <start> --end <end>'

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
            architectureType = arg
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
        elif opt in ("-c"):
            charges_count = int(arg)

    return baseFileName, gridSize, architectureType, count, epochs, startIndex, endIndex, charges_count, postfix


if __name__ == '__main__':

    baseFilename, gridSize, architectureType, count, epochs, startIndex, endIndex, charges_count, postfix = parseArguments(sys.argv[1:])

    s = ''
    for index in range(startIndex, endIndex+1):
        filename = make_filename(baseFilename, gridSize, architectureType, count, epochs, index, charges_count, postfix)

        with open(filename, "r") as read_file:
          data = json.load(read_file)

        avg_error_avg = data['avg_error_avg']
        avg_error_variance = data['avg_error_variance']

        learning_duration= data['learning_duration']

        #s = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(gridSize, count, architectureType, epochs, learning_duration, avg_error_avg,  avg_error_variance, variances_avg, variances_variance, median_values, median_values_variance, max_values, max_values_variance)
        #s1 = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t'.format(learning_duration, avg_error_avg,  avg_error_variance, variances_avg, variances_variance, median_values, median_values_variance, max_values, max_values_variance)
        #print(index, s1)

        s += '{}\t{}\t{}\t'.format(learning_duration, avg_error_avg,  avg_error_variance)

    print(s) # gridSize, count, architectureType, epochs, learning_duration, avg_error_avg,  avg_error_variance, variances_avg, variances_variance, median_values, median_values_variance, max_values, max_values_variance)
    #print(data)

