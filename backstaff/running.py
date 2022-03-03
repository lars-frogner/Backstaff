import subprocess


def run_command(command,
                return_immediately=False,
                logger=print,
                error_logger=print,
                **kwargs):
    process = subprocess.Popen(command,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               **kwargs)
    if isinstance(command, list):
        logger(subprocess.list2cmdline(process.args))
    else:
        logger(command)

    if return_immediately:
        return

    while True:
        output = process.stdout.readline().decode()
        error_msg = process.stderr.readline().decode()
        if output == '' and error_msg == '' and process.poll() is not None:
            break
        if output:
            logger(output.strip())
        if error_msg:
            error_logger(error_msg.strip())
    return_code = process.poll()
    return return_code


def run_backstaff(*args, pre_cargo_args=[], features=['cli'], **kwargs):
    args = pre_cargo_args + [
        'cargo', 'run', '--release', '--no-default-features', '--features',
        ' '.join(features), '--'
    ] + list(args)
    return run_command(args, **kwargs)


def run_backstaff_remotely(destination_machine,
                           destination_dir,
                           *args,
                           pre_ssh_args=[],
                           pre_cargo_args=['nice'],
                           features=['cli'],
                           **kwargs):
    run_command_args = pre_cargo_args + [
        'cargo', 'run', '--release', '--no-default-features', '--features',
        ' '.join(features), '--'
    ] + list(args)
    remote_command_args = ['cd', destination_dir, '&&', 'git', 'pull', '&&'
                           ] + run_command_args
    args = pre_ssh_args + [
        'ssh', destination_machine, ' '.join(remote_command_args)
    ]
    return run_command(args, **kwargs)
