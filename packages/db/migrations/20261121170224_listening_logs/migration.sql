CREATE TABLE `listening_logs` (
	`id` text PRIMARY KEY,
	`user_id` text NOT NULL,
	`song_isrc` text NOT NULL,
	`listened_on` text NOT NULL,
	`created_at` integer NOT NULL,
	CONSTRAINT `fk_listening_logs_user_id_users_id_fk` FOREIGN KEY (`user_id`) REFERENCES `users`(`id`) ON DELETE CASCADE,
	CONSTRAINT `fk_listening_logs_song_isrc_songs_isrc_fk` FOREIGN KEY (`song_isrc`) REFERENCES `songs`(`isrc`) ON DELETE RESTRICT,
	CONSTRAINT "listening_logs_listened_on_format" CHECK(length("listened_on") = 10 and "listened_on" glob '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]')
);
--> statement-breakpoint
CREATE INDEX `listening_logs_user_date_idx` ON `listening_logs` (`user_id`,`listened_on`,`created_at`);