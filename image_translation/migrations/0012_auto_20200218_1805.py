# Generated by Django 3.0.2 on 2020-02-18 10:05

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('image_translation', '0011_mengine_is_start'),
    ]

    operations = [
        migrations.AddField(
            model_name='train',
            name='create_time',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='mengine',
            name='create_time',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]
