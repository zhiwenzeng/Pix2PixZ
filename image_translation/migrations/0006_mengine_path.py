# Generated by Django 3.0.2 on 2020-02-05 14:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_translation', '0005_auto_20200205_0104'),
    ]

    operations = [
        migrations.AddField(
            model_name='mengine',
            name='path',
            field=models.CharField(max_length=200, null=True),
        ),
    ]
