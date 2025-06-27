#!/usr/bin/env python
import os
import sys
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mainfolder.settings')
django.setup()

from backend.models import Place

def populate_entry_fees():
    """
    Populate entry fees for popular places in Nepal based on actual tourist information.
    Fees are approximate and may vary. Sources: Nepal Tourism Board, travel websites, and local information.
    """
    
    # Dictionary of places with their entry fees
    # Format: {place_name: {tourist_fee_npr, tourist_fee_usd, local_fee_npr, saarc_fee_npr, fee_description}}
    entry_fees = {
        # Kathmandu Valley UNESCO Sites
        "Pashupatinath Temple": {
            "tourist_fee_npr": 1000,
            "tourist_fee_usd": 10,
            "local_fee_npr": 0,
            "saarc_fee_npr": 500,
            "fee_description": "Free for locals. Foreign tourists must pay. Photography restrictions apply in main temple area.",
            "has_entry_fee": True
        },
        "Swayambhunath Stupa": {
            "tourist_fee_npr": 200,
            "tourist_fee_usd": 2,
            "local_fee_npr": 0,
            "saarc_fee_npr": 100,
            "fee_description": "Free for locals. Small fee for foreigners. Best visited early morning or evening.",
            "has_entry_fee": True
        },
        "Boudhanath Stupa": {
            "tourist_fee_npr": 400,
            "tourist_fee_usd": 4,
            "local_fee_npr": 0,
            "saarc_fee_npr": 200,
            "fee_description": "Free for locals. Foreign tourists must pay. Circumambulation is free.",
            "has_entry_fee": True
        },
        "Kathmandu Durbar Square": {
            "tourist_fee_npr": 1000,
            "tourist_fee_usd": 10,
            "local_fee_npr": 0,
            "saarc_fee_npr": 500,
            "fee_description": "Part of UNESCO World Heritage Site. Free for locals. Includes access to museums.",
            "has_entry_fee": True
        },
        "Patan Durbar Square": {
            "tourist_fee_npr": 1000,
            "tourist_fee_usd": 10,
            "local_fee_npr": 0,
            "saarc_fee_npr": 500,
            "fee_description": "UNESCO World Heritage Site. Free for locals. Includes Patan Museum access.",
            "has_entry_fee": True
        },
        "Bhaktapur Durbar Square": {
            "tourist_fee_npr": 1500,
            "tourist_fee_usd": 15,
            "local_fee_npr": 0,
            "saarc_fee_npr": 750,
            "fee_description": "UNESCO World Heritage Site. Free for locals. Valid for entire old city area.",
            "has_entry_fee": True
        },
        "Changu Narayan Temple": {
            "tourist_fee_npr": 300,
            "tourist_fee_usd": 3,
            "local_fee_npr": 0,
            "saarc_fee_npr": 150,
            "fee_description": "Oldest temple in Kathmandu Valley. Free for locals. UNESCO site.",
            "has_entry_fee": True
        },
        
        # Religious Sites
        "Kopan Monastery": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Donations welcome. Meditation courses available for fee.",
            "has_entry_fee": False
        },
        "Golden Temple (Hiranya Varna Mahavihar)": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Donations welcome. Photography restrictions apply.",
            "has_entry_fee": False
        },
        "Dattatreya Temple": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Part of Bhaktapur Durbar Square complex.",
            "has_entry_fee": False
        },
        
        # Natural Sites & Parks
        "Shivapuri Nagarjun National Park": {
            "tourist_fee_npr": 100,
            "tourist_fee_usd": 1,
            "local_fee_npr": 50,
            "saarc_fee_npr": 75,
            "fee_description": "Entry fee for hiking. Different rates for locals and foreigners.",
            "has_entry_fee": True
        },
        "Phulchowki Hill": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Transportation costs apply. Best for bird watching.",
            "has_entry_fee": False
        },
        "Sundarijal": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Popular hiking destination. Waterfall and reservoir.",
            "has_entry_fee": False
        },
        "Langtang National Park": {
            "tourist_fee_npr": 3000,
            "tourist_fee_usd": 30,
            "local_fee_npr": 1500,
            "saarc_fee_npr": 1500,
            "fee_description": "National park entry fee. Required for trekking in Langtang region.",
            "has_entry_fee": True
        },
        
        # Cultural Sites
        "Kirtipur": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Ancient Newar town. Some temples may have small fees.",
            "has_entry_fee": False
        },
        "Madhyapur Thimi": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Traditional pottery town. Cultural heritage site.",
            "has_entry_fee": False
        },
        "Bungmati and Khokana": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Traditional Newar villages. Mustard oil production.",
            "has_entry_fee": False
        },
        "Chovar Hill": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Buddhist monastery and viewpoint. Historical significance.",
            "has_entry_fee": False
        },
        
        # Museums & Cultural Centers
        "National Museum": {
            "tourist_fee_npr": 150,
            "tourist_fee_usd": 1.5,
            "local_fee_npr": 50,
            "saarc_fee_npr": 75,
            "fee_description": "Reduced rates for locals and SAARC nationals. Photography fee extra.",
            "has_entry_fee": True
        },
        "Patan Museum": {
            "tourist_fee_npr": 500,
            "tourist_fee_usd": 5,
            "local_fee_npr": 100,
            "saarc_fee_npr": 250,
            "fee_description": "Part of Patan Durbar Square. Separate fee for museum access.",
            "has_entry_fee": True
        },
        
        # Other Popular Sites
        "Asan Market": {
            "tourist_fee_npr": 0,
            "tourist_fee_usd": 0,
            "local_fee_npr": 0,
            "saarc_fee_npr": 0,
            "fee_description": "Free entry. Traditional market. Best for local shopping experience.",
            "has_entry_fee": False
        },
        "Garden of Dreams": {
            "tourist_fee_npr": 200,
            "tourist_fee_usd": 2,
            "local_fee_npr": 50,
            "saarc_fee_npr": 100,
            "fee_description": "Historic garden. Reduced rates for locals. Cafe inside.",
            "has_entry_fee": True
        },
        "Narayanhiti Palace Museum": {
            "tourist_fee_npr": 500,
            "tourist_fee_usd": 5,
            "local_fee_npr": 100,
            "saarc_fee_npr": 250,
            "fee_description": "Former royal palace. Photography restrictions. Guided tours available.",
            "has_entry_fee": True
        }
    }
    
    updated_count = 0
    not_found_count = 0
    
    for place_name, fee_data in entry_fees.items():
        try:
            # Try to find the place by name (case insensitive)
            place = Place.objects.filter(name__icontains=place_name).first()
            
            if place:
                # Update the place with fee information
                place.tourist_fee_npr = fee_data.get("tourist_fee_npr")
                place.tourist_fee_usd = fee_data.get("tourist_fee_usd")
                place.local_fee_npr = fee_data.get("local_fee_npr")
                place.saarc_fee_npr = fee_data.get("saarc_fee_npr")
                place.fee_description = fee_data.get("fee_description", "")
                place.has_entry_fee = fee_data.get("has_entry_fee", False)
                place.save()
                
                print(f"✅ Updated: {place.name}")
                updated_count += 1
            else:
                print(f"❌ Not found: {place_name}")
                not_found_count += 1
                
        except Exception as e:
            print(f"❌ Error updating {place_name}: {str(e)}")
    
    print(f"\n📊 Summary:")
    print(f"✅ Successfully updated: {updated_count} places")
    print(f"❌ Not found: {not_found_count} places")
    print(f"📝 Total places in database: {Place.objects.count()}")

if __name__ == "__main__":
    populate_entry_fees() 