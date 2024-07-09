-- +goose Up
-- +goose StatementBegin
CREATE TABLE news(
    id SERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    image TEXT NULL,
    url TEXT NOT NULL UNIQUE,
    content TEXT NULL,
    date TIMESTAMP WITHOUT TIME ZONE,
    embedding VECTOR(768), -- Nomic AI embedding dims
    summary TEXT NULL,
    created_at TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW()
);
-- +goose StatementEnd

-- +goose Down
-- +goose StatementBegin
DROP TABLE news;
-- +goose StatementEnd
