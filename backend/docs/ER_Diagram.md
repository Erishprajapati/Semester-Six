# Entity Relationship Diagram Documentation

## System Overview
This document provides a detailed Entity Relationship (ER) diagram for the Tourism and Place Recommendation System.

## Database Schema

### Entities and Their Attributes

#### 1. User (Django Auth User)
- Primary Key: id
- username (String)
- email (String)
- password (String)
- first_name (String)
- last_name (String)
- is_active (Boolean)
- date_joined (DateTime)

#### 2. Place
- Primary Key: id
- name (String)
- description (Text)
- popular_for (Text)
- category (String)
- location (Text)
- district (String)
- latitude (Float)
- longitude (Float)
- image (ImageField)
- added_by (Foreign Key to User)

#### 3. Tag
- Primary Key: id
- name (String)

#### 4. CrowdData
- Primary Key: id
- place (Foreign Key to Place)
- timestamp (DateTime)
- crowdlevel (Integer)
- status (String) - Choices: ['High', 'Medium', 'Low']

#### 5. UserLocation
- Primary Key: id
- user (Foreign Key to User)
- latitude (Float)
- longitude (Float)
- created_at (DateTime)

#### 6. UserPreference
- Primary Key: id
- user (Foreign Key to User)

#### 7. SearchHistory
- Primary Key: id
- user (Foreign Key to User)
- search_query (String)
- search_type (String)
- timestamp (DateTime)

## Relationships

1. **User - Place**
   - Relationship Type: One-to-Many
   - Description: A user can add multiple places
   - Cardinality: 1:N

2. **Place - Tag**
   - Relationship Type: Many-to-Many
   - Description: A place can have multiple tags, and a tag can be associated with multiple places
   - Cardinality: M:N
   - Constraint: Maximum 4 tags per place

3. **Place - CrowdData**
   - Relationship Type: One-to-Many
   - Description: A place can have multiple crowd data entries
   - Cardinality: 1:N

4. **User - UserLocation**
   - Relationship Type: One-to-Many
   - Description: A user can have multiple location records
   - Cardinality: 1:N

5. **User - UserPreference**
   - Relationship Type: One-to-One
   - Description: Each user has one set of preferences
   - Cardinality: 1:1

6. **UserPreference - Tag**
   - Relationship Type: Many-to-Many
   - Description: User preferences can include multiple tags
   - Cardinality: M:N

7. **User - SearchHistory**
   - Relationship Type: One-to-Many
   - Description: A user can have multiple search history entries
   - Cardinality: 1:N

## Constraints

1. **Place**
   - Unique constraint on combination of name and district
   - Maximum of 4 tags per place

2. **CrowdData**
   - Status must be one of: 'High', 'Medium', 'Low'
   - Crowd level must be between 0 and 100

3. **SearchHistory**
   - Ordered by timestamp (most recent first)

## Indexes

1. Place
   - name
   - district
   - category

2. SearchHistory
   - timestamp
   - user_id

3. CrowdData
   - place_id
   - timestamp

## Notes

- All timestamps are automatically managed by Django
- Image uploads for places are stored in 'place_images/' directory
- User authentication is handled by Django's built-in auth system
- All foreign keys use CASCADE deletion except for Place's added_by (SET_NULL) 