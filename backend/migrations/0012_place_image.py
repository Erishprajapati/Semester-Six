# Generated by Django 5.2 on 2025-04-23 06:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('backend', '0011_alter_place_unique_together'),
    ]

    operations = [
        migrations.AddField(
            model_name='place',
            name='image',
            field=models.ImageField(blank=True, null=True, upload_to='place_images/'),
        ),
    ]
