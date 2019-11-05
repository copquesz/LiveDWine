-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema livedwine
-- -----------------------------------------------------
DROP SCHEMA IF EXISTS `livedwine` ;

-- -----------------------------------------------------
-- Schema livedwine
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `livedwine` DEFAULT CHARACTER SET utf8 ;
USE `livedwine` ;

-- -----------------------------------------------------
-- Table `livedwine`.`wine`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `livedwine`.`wine` (
  `wine_id` INT NOT NULL AUTO_INCREMENT,
  `wine_name` VARCHAR(255) NOT NULL,
  `type` VARCHAR(255) NOT NULL,
  `country` VARCHAR(255) NOT NULL,
  `region` VARCHAR(45) NOT NULL,
  `alcohol_content` DOUBLE NULL,
  `producer` VARCHAR(255) NULL,
  `service` INT NOT NULL,
  `volume` INT NOT NULL,
  `grape` VARCHAR(255) NULL,
  `harvest` INT NULL,
  `harmonization` TEXT NULL,
  `image` TEXT NULL,
  PRIMARY KEY (`wine_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

CREATE UNIQUE INDEX `wine_name_UNIQUE` ON `livedwine`.`wine` (`wine_name` ASC) VISIBLE;


-- -----------------------------------------------------
-- Table `livedwine`.`user`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `livedwine`.`user` (
  `user_id` INT NOT NULL AUTO_INCREMENT,
  `user_name` VARCHAR(255) NOT NULL,
  `gender` VARCHAR(255) NULL,
  `profession` VARCHAR(255) NULL,
  `age` INT NULL,
  PRIMARY KEY (`user_id`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;


-- -----------------------------------------------------
-- Table `livedwine`.`rating`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `livedwine`.`rating` (
  `wine_id` INT NOT NULL,
  `user_id` INT NOT NULL,
  `rating` DOUBLE NOT NULL,
  PRIMARY KEY (`wine_id`, `user_id`),
  CONSTRAINT `fk_wine_has_user_wine`
    FOREIGN KEY (`wine_id`)
    REFERENCES `livedwine`.`wine` (`wine_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fk_wine_has_user_user1`
    FOREIGN KEY (`user_id`)
    REFERENCES `livedwine`.`user` (`user_id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8;

CREATE INDEX `fk_wine_has_user_user1_idx` ON `livedwine`.`rating` (`user_id` ASC) VISIBLE;

CREATE INDEX `fk_wine_has_user_wine_idx` ON `livedwine`.`rating` (`wine_id` ASC) VISIBLE;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
