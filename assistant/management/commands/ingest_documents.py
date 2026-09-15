from django.core.management.base import BaseCommand

from assistant.services.rag_service import rag_service


class Command(BaseCommand):
    help = "Idempotently ingest UTF-8 text files into the assistant Chroma collection"

    def add_arguments(self, parser):
        parser.add_argument("files", nargs="+")

    def handle(self, *args, **options):
        for path in options["files"]:
            count = rag_service.add_documents_from_file(path)
            self.stdout.write(f"{path}: {count} stable chunk IDs upserted")
