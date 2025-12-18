from django.apps import AppConfig


class SpaceAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'space_app'

    def ready(self) -> None:
        # Runtime tmp layers are stored under repo ".tmp/" and should be cleaned only
        # when the server process starts/stops (not per-request).
        try:
            from .tmp_store import init_tmp_lifecycle_cleanup

            init_tmp_lifecycle_cleanup()
        except Exception:
            pass
