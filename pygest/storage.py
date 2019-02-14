import os
import boto3
from botocore.client import ClientError
from pygest.convenience import set_name


def can_list_buckets():
    s3 = boto3.resource('s3')
    try:
        s3.meta.client.head_bucket(Bucket="ms.mfs.ge-data")
        # It doesn't matter if our bucket is there or not, just that it didn't raise an exception.
        return True
    except ClientError:
        return False


def upload(args, logger=None):
    """
    Move the output files generated to external storage

    :param args: Original arguments dictionary from call of pygest binary
    :param logger: python3 logger instance
    :return: 0 if successful, otherwise a non-zero error code
    """

    base_path = set_name(args)
    if base_path[0:len(args.data)] == args.data:
        base_path = base_path[len(args.data) + 1:]
    if args.command in ["order", "push"]:
        files = [base_path + ".tsv", base_path + ".log", base_path + ".json"]
        if len(args.upload) > 0:
            if args.upload[0] == "s3":
                return upload_to_s3(files, args, logger)
            if args.upload[0] == "ssh":
                return upload_to_ssh(files, args, logger)
            if logger is not None:
                logger.warn("I cannot upload to {}. I can only do ssh or s3.".format(args.upload[0]))
            else:
                print("I cannot upload to {}. I can only do ssh or s3.".format(args.upload[0]))
            return 1
        else:
            # Do nothing, no upload targets specified.
            if logger is not None:
                logger.warn("Asked to upload, but no targets specified. Doing nothing.")
            else:
                print("Asked to upload, but no targets specified. Doing nothing.")
            return 0
    else:
        # Do nothing, I only upload with pushes and orders.
        if logger is not None:
            logger.warn("Asked to upload, but I only upload pushes or orders, not '{}'s".format(args.command))
        else:
            print("Asked to upload, but I only upload pushes or orders, not '{}'s".format(args.command))
        return 0


def upload_to_s3(files, args, logger):
    """

    :param files: a list of local file paths
    :param args: original command line arguments passed to pygest executable
    :param logger: a python logger object
    :return: 0 on success
    """

    # The --upload cmdline argument is --upload s3 bucket-name
    # args.upload[0] == "s3" and args.upload[1] provides the bucket name, if it exists.
    bucket_name = "ms.mfs.ge-data" if len(args.upload) < 2 else args.upload[1]

    s3 = boto3.Session().resource("s3")

    def bucket_holds_key(key_name):
        """ Return true if key is found, false if it's not, and raise an exception if we can't find out.
        """
        try:
            s3.Bucket(bucket_name).Object(key_name).load()
        except ClientError as ee:
            if ee.response['Error']['Code'] == '404':
                return False
            else:
                raise
        else:
            return True

    for f in files:
        print("Trying to upload {} to s3://{}/{}".format(os.path.join(args.data, f), bucket_name, f))
        f_local = os.path.join(args.data, f)
        try:
            if bucket_holds_key(f):
                logger.info("{} already exists in {}. Leaving it alone.".format(f, bucket_name))
            else:
                s3.Bucket(bucket_name).upload_file(f_local, f)
                # There is no json returned from this call. But an error raises an exception, so no news is good news.
        except ClientError as e:
            if e.response['Error']['Code'] == '403':
                # Permissions don't allow getting an object's HEAD
                logger.warn("You are not allowed to even check if a file exists in bucket ({}).".format(bucket_name))
                logger.warn("Check your [default] key in ~/.aws/credentials, and verify AWS IAM permissions.")
                break
            else:
                raise

    return 0


def upload_to_ssh(obj_path, args, logger):
    """

    :param obj_path:
    :param args:
    :param logger:
    :return:
    """

    logger.warn("Uploading to ssh is not yet implemented. You'll have to scp {} from the command line.".format(
        os.path.join(args.data, obj_path)
    ))
    return 0


def test_upload_results_bad_bucket():
    pass


def test_upload_results_file_exists():
    pass


def test_upload_results_new_file():
    pass


def check_aws(args):
    """
    Determine permissions of existing aws IAM keys, and if possible the state of the data there

    :param args: original command-line arguments from pygest executable
    :return: 0 for success, integer error codes for failure
    """

    print("Testing AWS configuration and security...")
    # The --upload cmdline argument is --upload s3 bucket-name
    # args.upload[0] == "s3" and args.upload[1] provides the bucket name, if it exists.
    if len(args.upload) > 1:
        bucket_name = args.upload[1]
        print("  Using bucket {} as specified in --upload argument".format(bucket_name))
    else:
        bucket_name = "ms.mfs.ge-data"
        print("  No bucket specified, using the default, {}".format(bucket_name))

    s3 = boto3.Session().resource("s3")

    # See if we can list buckets, and that list includes what we're looking for.
    try:
        if s3.Bucket(bucket_name) in s3.buckets.all():
            print("  You can list buckets, and {} is accessible.".format(bucket_name))
        else:
            print("  You can list buckets, but {} is not accessible.".format(bucket_name))
            print("    check ~/.aws/credentials; PyGEST is only capable of using the [default] credentials.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'AccessDenied':
            print("  You do not have permissions to list buckets.")
            print("    check ~/.aws/credentials; PyGEST is only capable of using the [default] credentials.")
        else:
            print("  ERROR ({}) listing buckets. {}".format(
                e.response['Error']['Code'], e.response['Error']['Message']
            ))

    # See if we can read the contents of a file
    try:
        participants = s3.Object(bucket_name, "sourcedata/participants.tsv").get()['Body'].read().decode('utf-8')
        if len(participants) > 0:
            print("  You can read objects from the {} bucket.".format(bucket_name))
        else:
            print("  No exceptions were raised, but the test file came back empty.")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print("  Files known to exist in a valid dataset (sourcedata/participants.tsv) don't in {}".format(
                 bucket_name
            ))
        else:
            print("  ERROR ({}) reading an object. {}".format(
                e.response['Error']['Code'], e.response['Error']['Message']
            ))

    # See if we can write a dummy file
    try:
        response = s3.Object(bucket_name, "dummyfile.deletable").put(Body=b"testing")
        if response['ResponseMetadata']['HTTPStatusCode'] == 200:
            print("  You can write objects to the {} bucket.".format(bucket_name))
        else:
            print("  Got http status {} trying to write a file to {}, but no exceptions were raised.".format(
                response['ResponseMedadata']['HTTPStatusCode'], bucket_name
            ))
    except ClientError as e:
        print("  ERROR ({}) writing to bucket. {}".format(
            e.response['Error']['Code'], e.response['Error']['Message']
        ))

    return 0
