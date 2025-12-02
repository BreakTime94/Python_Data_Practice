import strawberry

@strawberry.type
class Mutation:
    @strawberry.mutation
    def ping(self) -> str:
        """
        동작 확인용 Mutation (실제 DB 변경 없음)
        """
        return "pong"