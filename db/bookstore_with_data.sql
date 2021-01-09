/*
 Navicat Premium Data Transfer

 Source Server         : localhost_3306
 Source Server Type    : MySQL
 Source Server Version : 80019
 Source Host           : localhost:3306
 Source Schema         : bookstore

 Target Server Type    : MySQL
 Target Server Version : 80019
 File Encoding         : 65001

 Date: 09/01/2021 21:21:24
*/

SET NAMES utf8;
SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for book
-- ----------------------------
DROP TABLE IF EXISTS `book`;
CREATE TABLE `book`  (
  `book_id` int(0) NOT NULL AUTO_INCREMENT COMMENT '图书唯一标识符',
  `title` varchar(45) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '图书名称',
  `author` varchar(45) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '图书作者',
  `isbn` varchar(45) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '图书的ISBN',
  `count` int(0) NOT NULL DEFAULT 0 COMMENT '图书在库数量',
  `price` decimal(10, 2) NOT NULL COMMENT '图书单价',
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) COMMENT '创建时间',
  `updated_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) ON UPDATE CURRENT_TIMESTAMP(0) COMMENT '更新时间',
  PRIMARY KEY (`book_id`) USING BTREE,
  UNIQUE INDEX `isbn_UNIQUE`(`isbn`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 11 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of book
-- ----------------------------
INSERT INTO `book` VALUES (1, 'War and peace', 'Leo Tolstoy', '0000', 5, 25.80, '2021-01-09 13:43:18', '2021-01-09 21:21:06');
INSERT INTO `book` VALUES (2, 'Notre Dame de Paris', 'Hugo', '3249', 529, 34.20, '2021-01-09 13:43:42', '2021-01-09 21:21:06');
INSERT INTO `book` VALUES (3, 'Childhood', 'Gorky', '0834', 7, 9.80, '2021-01-09 13:44:15', '2021-01-09 21:21:06');
INSERT INTO `book` VALUES (4, 'Wuthering HeightsEmily Bronte', 'Emily Bronte', '2391', 86, 65.10, '2021-01-09 13:44:54', '2021-01-09 21:21:06');
INSERT INTO `book` VALUES (5, 'David Copperfield', 'Charles Dickens', '0018', 19, 17.20, '2021-01-09 13:45:20', '2021-01-09 21:21:06');
INSERT INTO `book` VALUES (6, 'Red and Black', 'Stendhal', '1239', 13, 41.50, '2021-01-09 13:45:44', '2021-01-09 21:21:06');
INSERT INTO `book` VALUES (7, 'Les Miserables', 'Hugo', '0012', 17, 32.80, '2021-01-09 13:46:14', '2021-01-09 21:21:06');
INSERT INTO `book` VALUES (8, 'Anna Karenina', 'Leo Tolstoy', '0630', 58, 99.10, '2021-01-09 13:46:45', '2021-01-09 21:21:06');
INSERT INTO `book` VALUES (9, 'John Christopher', 'Roman Roland', '6292', 49, 63.20, '2021-01-09 13:47:21', '2021-01-09 21:21:07');
INSERT INTO `book` VALUES (10, 'Gone with the Wind', 'Margaret Mitchell', '2311', 111, 523.30, '2021-01-09 13:47:47', '2021-01-09 13:47:47');

-- ----------------------------
-- Table structure for offer
-- ----------------------------
DROP TABLE IF EXISTS `offer`;
CREATE TABLE `offer`  (
  `offer_id` int(0) NOT NULL AUTO_INCREMENT COMMENT '供货单（未成交）ID',
  `provider_id` int(0) NOT NULL COMMENT '供应商ID',
  `book_id` int(0) NOT NULL COMMENT '图书ID',
  `price` decimal(10, 2) NOT NULL COMMENT '供应图书单价',
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) COMMENT '创建时间',
  `updated_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) ON UPDATE CURRENT_TIMESTAMP(0) COMMENT '修改时间',
  PRIMARY KEY (`offer_id`) USING BTREE,
  INDEX `book_id_idx`(`book_id`) USING BTREE,
  INDEX `provider_id_idx`(`provider_id`) USING BTREE,
  CONSTRAINT `book_id` FOREIGN KEY (`book_id`) REFERENCES `book` (`book_id`) ON DELETE RESTRICT ON UPDATE RESTRICT,
  CONSTRAINT `provider_id` FOREIGN KEY (`provider_id`) REFERENCES `provider` (`provider_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 71 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of offer
-- ----------------------------
INSERT INTO `offer` VALUES (1, 1, 1, 23.22, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (2, 1, 2, 30.78, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (3, 1, 3, 8.82, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (4, 1, 4, 58.59, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (5, 1, 5, 15.48, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (6, 1, 6, 37.35, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (7, 1, 7, 29.52, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (8, 1, 8, 89.19, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (9, 1, 9, 56.88, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (10, 1, 10, 470.97, '2021-01-09 13:53:25', '2021-01-09 13:53:25');
INSERT INTO `offer` VALUES (16, 2, 1, 16.77, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (17, 2, 2, 22.23, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (18, 2, 3, 6.37, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (19, 2, 4, 42.32, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (20, 2, 5, 11.18, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (21, 2, 6, 26.98, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (22, 2, 7, 21.32, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (23, 2, 8, 64.42, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (24, 2, 9, 41.08, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (25, 2, 10, 340.15, '2021-01-09 13:53:42', '2021-01-09 13:53:42');
INSERT INTO `offer` VALUES (31, 3, 1, 30.96, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (32, 3, 2, 41.04, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (33, 3, 3, 11.76, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (34, 3, 4, 78.12, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (35, 3, 5, 20.64, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (36, 3, 6, 49.80, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (37, 3, 7, 39.36, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (38, 3, 8, 118.92, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (39, 3, 9, 75.84, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (40, 3, 10, 627.96, '2021-01-09 13:53:58', '2021-01-09 13:53:58');
INSERT INTO `offer` VALUES (46, 4, 1, 22.70, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (47, 4, 2, 30.10, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (48, 4, 3, 8.62, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (49, 4, 4, 57.29, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (50, 4, 5, 15.14, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (51, 4, 6, 36.52, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (52, 4, 7, 28.86, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (53, 4, 8, 87.21, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (54, 4, 9, 55.62, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (55, 4, 10, 460.50, '2021-01-09 13:54:06', '2021-01-09 13:54:06');
INSERT INTO `offer` VALUES (61, 5, 1, 51.60, '2021-01-09 13:54:12', '2021-01-09 13:54:12');
INSERT INTO `offer` VALUES (62, 5, 2, 68.40, '2021-01-09 13:54:12', '2021-01-09 13:54:12');
INSERT INTO `offer` VALUES (63, 5, 3, 19.60, '2021-01-09 13:54:12', '2021-01-09 13:54:12');
INSERT INTO `offer` VALUES (64, 5, 4, 130.20, '2021-01-09 13:54:12', '2021-01-09 13:54:12');
INSERT INTO `offer` VALUES (65, 5, 5, 34.40, '2021-01-09 13:54:12', '2021-01-09 13:54:12');
INSERT INTO `offer` VALUES (66, 5, 6, 83.00, '2021-01-09 13:54:12', '2021-01-09 13:54:12');
INSERT INTO `offer` VALUES (67, 5, 7, 65.60, '2021-01-09 13:54:12', '2021-01-09 13:54:12');
INSERT INTO `offer` VALUES (68, 5, 8, 198.20, '2021-01-09 13:54:12', '2021-01-09 13:54:12');
INSERT INTO `offer` VALUES (69, 5, 9, 126.40, '2021-01-09 13:54:12', '2021-01-09 13:54:12');
INSERT INTO `offer` VALUES (70, 5, 10, 1046.60, '2021-01-09 13:54:12', '2021-01-09 13:54:12');

-- ----------------------------
-- Table structure for provider
-- ----------------------------
DROP TABLE IF EXISTS `provider`;
CREATE TABLE `provider`  (
  `provider_id` int(0) NOT NULL AUTO_INCREMENT COMMENT '供应商ID',
  `name` varchar(45) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '名字',
  `phone` varchar(45) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '联系电话',
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) COMMENT '创建时间',
  `updated_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) ON UPDATE CURRENT_TIMESTAMP(0) COMMENT '更新时间',
  PRIMARY KEY (`provider_id`) USING BTREE
) ENGINE = InnoDB AUTO_INCREMENT = 6 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of provider
-- ----------------------------
INSERT INTO `provider` VALUES (1, 'Zhongshan University Press', '1001111', '2021-01-09 13:49:28', '2021-01-09 13:50:38');
INSERT INTO `provider` VALUES (2, 'Guangdong People\'s Publishing House', '2120901', '2021-01-09 13:49:46', '2021-01-09 13:49:46');
INSERT INTO `provider` VALUES (3, 'Guangdong Education Press', '9990295', '2021-01-09 13:50:01', '2021-01-09 13:50:01');
INSERT INTO `provider` VALUES (4, 'Guangdong Science and Technology Press', '7220194', '2021-01-09 13:50:12', '2021-01-09 13:50:12');
INSERT INTO `provider` VALUES (5, 'Guangdong Higher Education Press', '3402932', '2021-01-09 13:50:22', '2021-01-09 13:50:22');

-- ----------------------------
-- Table structure for purchase
-- ----------------------------
DROP TABLE IF EXISTS `purchase`;
CREATE TABLE `purchase`  (
  `order_id` int(0) NOT NULL AUTO_INCREMENT COMMENT '购书订单（消费者）ID',
  `book_id` int(0) NOT NULL COMMENT '图书ID',
  `price` decimal(10, 2) NOT NULL COMMENT '购买时单价',
  `count` int(0) NOT NULL DEFAULT 0 COMMENT '购买数量',
  `customer_name` varchar(45) CHARACTER SET utf8 COLLATE utf8_general_ci NOT NULL COMMENT '消费者名称',
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) COMMENT '创建时间',
  `updated_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) ON UPDATE CURRENT_TIMESTAMP(0) COMMENT '修改时间',
  PRIMARY KEY (`order_id`) USING BTREE,
  INDEX `order_book_id_idx`(`book_id`) USING BTREE,
  CONSTRAINT `purchase_order_book_id` FOREIGN KEY (`book_id`) REFERENCES `book` (`book_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB AUTO_INCREMENT = 10 CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of purchase
-- ----------------------------
INSERT INTO `purchase` VALUES (1, 1, 25.80, 1, 'zhenggehan', '2021-01-09 21:13:27', '2021-01-09 21:13:27');
INSERT INTO `purchase` VALUES (2, 2, 34.20, 1, 'zhenggehan', '2021-01-09 21:13:27', '2021-01-09 21:13:27');
INSERT INTO `purchase` VALUES (3, 3, 9.80, 1, 'zhenggehan', '2021-01-09 21:13:27', '2021-01-09 21:13:27');
INSERT INTO `purchase` VALUES (4, 4, 65.10, 1, 'zhenggehan', '2021-01-09 21:13:27', '2021-01-09 21:13:27');
INSERT INTO `purchase` VALUES (5, 5, 17.20, 1, 'zhenggehan', '2021-01-09 21:13:27', '2021-01-09 21:13:27');
INSERT INTO `purchase` VALUES (6, 6, 41.50, 1, 'zhenggehan', '2021-01-09 21:13:27', '2021-01-09 21:13:27');
INSERT INTO `purchase` VALUES (7, 7, 32.80, 1, 'zhenggehan', '2021-01-09 21:13:27', '2021-01-09 21:13:27');
INSERT INTO `purchase` VALUES (8, 8, 99.10, 1, 'zhenggehan', '2021-01-09 21:13:28', '2021-01-09 21:13:28');
INSERT INTO `purchase` VALUES (9, 9, 63.20, 1, 'zhenggehan', '2021-01-09 21:13:28', '2021-01-09 21:13:28');
INSERT INTO `purchase` VALUES (10, 1, 25.80, 1, 'zhenggehan', '2021-01-09 21:21:06', '2021-01-09 21:21:06');
INSERT INTO `purchase` VALUES (11, 2, 34.20, 1, 'zhenggehan', '2021-01-09 21:21:06', '2021-01-09 21:21:06');
INSERT INTO `purchase` VALUES (12, 3, 9.80, 1, 'zhenggehan', '2021-01-09 21:21:06', '2021-01-09 21:21:06');
INSERT INTO `purchase` VALUES (13, 4, 65.10, 1, 'zhenggehan', '2021-01-09 21:21:06', '2021-01-09 21:21:06');
INSERT INTO `purchase` VALUES (14, 5, 17.20, 1, 'zhenggehan', '2021-01-09 21:21:06', '2021-01-09 21:21:06');
INSERT INTO `purchase` VALUES (15, 6, 41.50, 1, 'zhenggehan', '2021-01-09 21:21:06', '2021-01-09 21:21:06');
INSERT INTO `purchase` VALUES (16, 7, 32.80, 1, 'zhenggehan', '2021-01-09 21:21:06', '2021-01-09 21:21:06');
INSERT INTO `purchase` VALUES (17, 8, 99.10, 1, 'zhenggehan', '2021-01-09 21:21:06', '2021-01-09 21:21:06');
INSERT INTO `purchase` VALUES (18, 9, 63.20, 1, 'zhenggehan', '2021-01-09 21:21:07', '2021-01-09 21:21:07');

-- ----------------------------
-- Table structure for refund
-- ----------------------------
DROP TABLE IF EXISTS `refund`;
CREATE TABLE `refund`  (
  `refund_id` int(0) NOT NULL AUTO_INCREMENT COMMENT '退款单ID',
  `order_id` int(0) NOT NULL COMMENT '对应的订单ID',
  `count` int(0) NOT NULL COMMENT '退货数量',
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) COMMENT '创建时间',
  `updated_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) ON UPDATE CURRENT_TIMESTAMP(0) COMMENT '更新时间',
  PRIMARY KEY (`refund_id`) USING BTREE,
  INDEX `order_id_idx`(`order_id`) USING BTREE,
  CONSTRAINT `order_id` FOREIGN KEY (`order_id`) REFERENCES `purchase` (`order_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of refund
-- ----------------------------

-- ----------------------------
-- Table structure for stock
-- ----------------------------
DROP TABLE IF EXISTS `stock`;
CREATE TABLE `stock`  (
  `stock_id` int(0) NOT NULL AUTO_INCREMENT COMMENT '购入（入库）单ID',
  `offer_id` int(0) NOT NULL COMMENT '对应的供应单ID',
  `count` int(0) NOT NULL COMMENT '购入图书数量',
  `created_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) COMMENT '创建时间',
  `updated_at` datetime(0) NOT NULL DEFAULT CURRENT_TIMESTAMP(0) ON UPDATE CURRENT_TIMESTAMP(0) COMMENT '更新时间',
  PRIMARY KEY (`stock_id`) USING BTREE,
  INDEX `offer_id_idx`(`offer_id`) USING BTREE,
  CONSTRAINT `offer_id` FOREIGN KEY (`offer_id`) REFERENCES `offer` (`offer_id`) ON DELETE RESTRICT ON UPDATE RESTRICT
) ENGINE = InnoDB CHARACTER SET = utf8 COLLATE = utf8_general_ci ROW_FORMAT = Dynamic;

-- ----------------------------
-- Records of stock
-- ----------------------------

-- ----------------------------
-- Procedure structure for purchase
-- ----------------------------
DROP PROCEDURE IF EXISTS `purchase`;
delimiter ;;
CREATE PROCEDURE `purchase`(IN id INT, IN purchaseCnt INT, IN customer_name VARCHAR(45))
BEGIN
	DECLARE c INT;
    declare total_price decimal(10,2);
    start transaction;
    select count into c FROM book WHERE book_id=id;
    if (c>=purchaseCnt) then
        SELECT book.*,price*purchaseCnt as total_cost FROM book WHERE book_id=id;
        UPDATE book set count=count-purchaseCnt where book_id=id;
        insert into bookstore.purchase select null,id, price,purchaseCnt, customer_name,now(),now() FROM book WHERE book_id=id;
    else
		SELECT book.*,price*purchaseCnt as total_cost FROM book WHERE book_id=-1;
	end if;
    commit;
END
;;
delimiter ;

SET FOREIGN_KEY_CHECKS = 1;
