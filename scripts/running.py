import subprocess


def run_bifrost_rust(*args, features=['cli']):
    args = [
        'cargo', 'run', '--release', '--no-default-features', '--features',
        ' '.join(features), '--'
    ] + list(args)
    print(' '.join(args))
    process = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    while True:
        output = process.stdout.readline().decode()
        error_msg = process.stderr.readline().decode()
        if output == '' and error_msg == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
        if error_msg:
            print(error_msg.strip())
    return_code = process.poll()
    return return_code
