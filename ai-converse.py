import boto3


def create_bedrock_client(region='us-west-2'):
    return boto3.client(service_name='bedrock-runtime', region_name=region)


def create_model_config(model_id, system_text):
    return {'modelId': model_id, 'system': [{'text': system_text}]}


def create_initial_message(text, role='user'):
    return {'role': role, 'content': [{'text': text}]}


def converse_with_model_stream(client, model_config, messages):
    response = client.converse_stream(
        modelId=model_config['modelId'],
        system=model_config['system'],
        messages=messages,
    )

    full_response = ""
    for event in response['stream']:
        if 'contentBlockDelta' in event:
            if 'text' in event['contentBlockDelta']['delta']:
                text_chunk = event['contentBlockDelta']['delta']['text']
                full_response += text_chunk
                print(text_chunk, end='', flush=True)

    print()  # 改行を追加
    return {'role': 'assistant', 'content': [{'text': full_response}]}


def print_speaker(speaker):
    print(f'{speaker}: ', end='', flush=True)


def swap_role(message, new_role):
    new_message = message.copy()
    new_message['role'] = new_role
    return new_message


def main():
    brt = create_bedrock_client()

    claude = create_model_config(
        'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
        'あなたは Mistral の壁打ち役である Claudeです。Mistral の企画した内容をいろんな角度からヒアリングして企画を完成に導いてください。',
    )

    mistral = create_model_config(
        'mistral.mistral-large-2407-v1:0',
        'あなたは Mistral という企画立案者です。Claude という壁打ち相手がいるので、企画の素案を持ち込んで、企画を完成させてください。',
    )

    seed_text = '今日は新しいチャンネル登録者数 100万人を超える YouTuber の案を思いついたんだって？'
    print('Claude: ' + seed_text)

    mistral_messages = [create_initial_message(seed_text)]
    claude_messages = [
        create_initial_message('{相手の発言待ち}'),
        create_initial_message(seed_text, 'assistant'),
    ]

    for _ in range(10):
        print_speaker('-------------Mistral-------------\n')
        mistral_response = converse_with_model_stream(brt, mistral, mistral_messages)
        mistral_messages.append(mistral_response)
        claude_messages.append(swap_role(mistral_response, 'user'))

        print_speaker('-------------Claude-------------\n')
        claude_response = converse_with_model_stream(brt, claude, claude_messages)
        claude_messages.append(claude_response)
        mistral_messages.append(swap_role(claude_response, 'user'))


if __name__ == "__main__":
    main()
