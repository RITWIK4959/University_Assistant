#!/usr/bin/env python3
"""
Monkey patch for LiveKit agents to fix aiohttp ClientSession proxy issue
"""

import aiohttp
from typing import Optional

# Store the original ClientSession.__init__
_original_client_session_init = aiohttp.ClientSession.__init__

def patched_client_session_init(self, *args, **kwargs):
    """
    Patched ClientSession.__init__ that removes the 'proxy' parameter
    and uses trust_env=True instead for proxy configuration
    """
    # Remove the proxy parameter if it exists
    proxy = kwargs.pop('proxy', None)
    
    # If proxy was specified, enable trust_env to use environment proxy settings
    if proxy is not None:
        kwargs['trust_env'] = True
    
    # Call the original __init__ with modified kwargs
    return _original_client_session_init(self, *args, **kwargs)

def apply_livekit_patch():
    """
    Apply the monkey patch to fix LiveKit aiohttp compatibility
    """
    print("🔧 Applying LiveKit aiohttp compatibility patch...")
    aiohttp.ClientSession.__init__ = patched_client_session_init
    print("✅ Patch applied successfully!")

def remove_livekit_patch():
    """
    Remove the monkey patch (restore original behavior)
    """
    print("🔄 Removing LiveKit patch...")
    aiohttp.ClientSession.__init__ = _original_client_session_init
    print("✅ Original behavior restored!")

if __name__ == "__main__":
    print("LiveKit aiohttp compatibility patch module")
    print("Import this module and call apply_livekit_patch() before using LiveKit agents")