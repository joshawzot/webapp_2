import boto3
from boto3.dynamodb.conditions import Key

# Initialize a DynamoDB service resource and specify the region
dynamodb = boto3.resource('dynamodb', region_name='us-east-2')

# Assuming you have a table named 'test'
table = dynamodb.Table('test')

def get_items():
    # Get all items from the table
    response = table.scan()
    return response['Items']

def add_item(item_data):
    # Add an item to the table
    table.put_item(Item=item_data)