# Generated by Django 3.0.2 on 2020-02-16 09:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('image_translation', '0009_auto_20200216_1352'),
    ]

    operations = [
        migrations.AddField(
            model_name='mengine',
            name='msg',
            field=models.CharField(max_length=200, null=True),
        ),
    ]