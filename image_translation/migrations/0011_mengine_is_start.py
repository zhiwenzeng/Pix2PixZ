# Generated by Django 3.0.2 on 2020-02-18 08:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_translation', '0010_mengine_msg'),
    ]

    operations = [
        migrations.AddField(
            model_name='mengine',
            name='is_start',
            field=models.BooleanField(default=False),
        ),
    ]
