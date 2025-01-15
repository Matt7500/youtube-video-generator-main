import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from settings import get_channel_names

def get_profile_names():
    # Update to use MongoDB function
    return get_channel_names('229202')  # Replace with dynamic username

def get_authenticated_credentials(profile_name, client_secrets_file, scopes):
    flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
    # Try different ports if 8080 is unavailable
    for port in range(8080, 8090):
        try:
            credentials = flow.run_local_server(port=port)
            break
        except OSError:
            if port == 8089:  # Last attempt
                raise Exception("Unable to find an available port for authentication")
            continue

    creds_data = {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

    os.makedirs('credentials', exist_ok=True)
    creds_file = f'credentials/youtube_credentials_{profile_name}.json'
    with open(creds_file, 'w') as f:
        json.dump(creds_data, f)

    print(f"Credentials saved to '{creds_file}'")
    return creds_data


def authenticate_profile(profile_name, client_secrets_file, scopes):
    creds_file = f'credentials/youtube_credentials_{profile_name}.json'
    if os.path.exists(creds_file):
        print(f"Existing credentials found for profile '{profile_name}'. Do you want to generate new credentials?")
        response = input("Enter 'yes' to generate new credentials, or any other key to keep existing ones: ")
        if response.lower() != 'yes':
            print(f"Keeping existing credentials for '{profile_name}'.")
            return

    print(f"Authenticating profile: {profile_name}")
    get_authenticated_credentials(profile_name, client_secrets_file, scopes)
    print(f"Authentication successful for profile: {profile_name}")


def main(username, profile=None):
    # Get profiles from MongoDB
    profiles = get_channel_names(username)
    
    # These values should come from MongoDB user settings
    client_secrets_file = "client_secret1.json"
    scopes = ['https://www.googleapis.com/auth/youtube.upload']

    if profile:
        if profile in profiles:
            authenticate_profile(profile, client_secrets_file, scopes)
        else:
            print(f"Error: Profile '{profile}' not found for user {username}")
    else:
        print("Available profiles:", ", ".join(profiles))
        for profile in profiles:
            print(f"\nProcessing profile: {profile}")
            authenticate_profile(profile, client_secrets_file, scopes)

    print("\nAuthentication process completed.")


if __name__ == '__main__':
    # Example usage:
    main(username="229202", profile=None)
