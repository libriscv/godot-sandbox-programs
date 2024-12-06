use autosurgeon::{Reconcile, Hydrate, hydrate, reconcile};
use automerge::AutoCommit;
use std::collections::HashMap;

#[derive(Debug, Clone, Reconcile, Hydrate, PartialEq)]
enum Variant {
    Nil,
    Bool(bool),
    Int(i64),
    Float(f64),
    String(String),
    Vector2(f64, f64),
    Vector3(f64, f64, f64),
    Vector2i(i64, i64),
    Vector3i(i64, i64, i64),
    Rect2(f64, f64, f64, f64),
    Rect2i(i64, i64, i64, i64),
    Vector4(f64, f64, f64, f64),
    Vector4i(i64, i64, i64, i64),
    Plane(f64, f64, f64, f64),
    Quaternion(f64, f64, f64, f64),
    AABB(Vector3, Vector3),
    Basis(Vector3, Vector3, Vector3),
    Transform2D(f64, f64, f64, f64, f64, f64),
    Transform3D(Vector3, Vector3, Vector3, Vector3),
    Projection(Vector4, Vector4, Vector4, Vector4),
    Color(f64, f64, f64, f64),
    Dictionary(HashMap<String, Variant>),
    Array(Vec<Variant>),
    // Add other types as needed
}

type Contact = Variant;
type Address = Variant;

fn main() {
    let mut contact = Contact {
        "name".to_string(): Variant::String("Sherlock Holmes".to_string()),
        "address".to_string(): Variant::String("221B Baker St".to_string()),
        "city".to_string(): Variant::String("London".to_string()),
        "postcode".to_string(): Variant::String("NW1 6XE".to_string()),
    };

    // Put data into a document
    let mut doc = AutoCommit::new();
    reconcile(&mut doc, &contact).unwrap();

    // Get data out of a document
    let contact2: Contact = hydrate(&doc).unwrap();
    assert_eq!(contact, contact2);

    // Fork and make changes
    let mut doc2 = doc.fork().with_actor(automerge::ActorId::random());
    let mut contact2: Contact = hydrate(&doc2).unwrap();
    contact2.insert("name".to_string(), Variant::String("Dangermouse".to_string()));
    reconcile(&mut doc2, &contact2).unwrap();

    // Concurrently on doc1
    contact.insert("address".to_string(), Variant::String("221C Baker St".to_string()));
    reconcile(&mut doc, &contact).unwrap();

    // Now merge the documents
    doc.merge(&mut doc2).unwrap();

    let merged: Contact = hydrate(&doc).unwrap();
    assert_eq!(merged, Contact {
        "name".to_string(): Variant::String("Dangermouse".to_string()), // This was updated in the first doc
        "address".to_string(): Variant::String("221C Baker St".to_string()), // This was concurrently updated in doc2
        "city".to_string(): Variant::String("London".to_string()),
        "postcode".to_string(): Variant::String("NW1 6XE".to_string()),
    });

    let contact1 = Contact {
        "name".to_string(): Variant::String("Sherlock Holmes".to_string()),
        "address".to_string(): Variant::String("221B Baker St".to_string()),
        "city".to_string(): Variant::String("London".to_string()),
        "postcode".to_string(): Variant::String("NW1 6XE".to_string()),
    };

    let contact2 = Contact {
        "name".to_string(): Variant::String("Dangermouse".to_string()),
        "address".to_string(): Variant::String("221C Baker St".to_string()),
        "city".to_string(): Variant::String("London".to_string()),
        "postcode".to_string(): Variant::String("NW1 6XE".to_string()),
    };

    let merged_contact = automerge(vec![contact1, contact2]);
    println!("{:?}", merged_contact);

    let variant = Variant::Vector2(1.0, 2.0);

    let mut doc = AutoCommit::new();
    reconcile(&mut doc, &variant).unwrap();

    let hydrated_variant: Variant = hydrate(&doc).unwrap();
    println!("{:?}", hydrated_variant);
}

fn automerge(variants: Vec<Contact>) -> Contact {
    let mut docs: Vec<AutoCommit> = variants.iter().map(|contact| {
        let mut doc = AutoCommit::new();
        reconcile(&mut doc, contact).unwrap();
        doc
    }).collect();

    let mut base_doc = docs.remove(0);
    for doc in docs {
        base_doc.merge(&mut doc.fork()).unwrap();
    }

    hydrate(&base_doc).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_variant_json_conversion() {
        fn test<T: serde::Serialize + for<'de> serde::Deserialize<'de> + PartialEq + std::fmt::Debug>(value: T, expected_json: serde_json::Value) {
            let serialized = serde_json::to_value(&value).unwrap();
            assert_eq!(serialized, expected_json);

            let deserialized: T = serde_json::from_value(serialized).unwrap();
            assert_eq!(deserialized, value);
        }

        // `Nil` and `bool` (represented as JSON keyword literals).
        test(Variant::Nil, json!(null));
        test(Variant::Bool(false), json!(false));
        test(Variant::Bool(true), json!(true));

        // Numbers and strings (represented as JSON strings).
        test(Variant::Int(1), json!("i:1"));
        test(Variant::Float(1.0), json!("f:1.0"));
        test(Variant::String("abc".to_string()), json!("s:abc"));

        // Math types.
        test(Variant::Vector2(1.0, 2.0), json!({"type": "Vector2", "args": [1.0, 2.0]}));
        test(Variant::Vector3(1.0, 2.0, 3.0), json!({"type": "Vector3", "args": [1.0, 2.0, 3.0]}));
        test(Variant::Vector2i(1, 2), json!({"type": "Vector2i", "args": [1, 2]}));
        test(Variant::Vector3i(1, 2, 3), json!({"type": "Vector3i", "args": [1, 2, 3]}));
        test(Variant::Rect2(1.0, 2.0, 3.0, 4.0), json!({"type": "Rect2", "args": [1.0, 2.0, 3.0, 4.0]}));
        test(Variant::Rect2i(1, 2, 3, 4), json!({"type": "Rect2i", "args": [1, 2, 3, 4]}));
        test(Variant::Vector4(1.0, 2.0, 3.0, 4.0), json!({"type": "Vector4", "args": [1.0, 2.0, 3.0, 4.0]}));
        test(Variant::Vector4i(1, 2, 3, 4), json!({"type": "Vector4i", "args": [1, 2, 3, 4]}));
        test(Variant::Plane(1.0, 2.0, 3.0, 4.0), json!({"type": "Plane", "args": [1.0, 2.0, 3.0, 4.0]}));
        test(Variant::Quaternion(1.0, 2.0, 3.0, 4.0), json!({"type": "Quaternion", "args": [1.0, 2.0, 3.0, 4.0]}));
        test(Variant::AABB(Variant::Vector3(1.0, 2.0, 3.0), Variant::Vector3(4.0, 5.0, 6.0)), json!({"type": "AABB", "args": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}));
        test(Variant::Basis(Variant::Vector3(1.0, 2.0, 3.0), Variant::Vector3(4.0, 5.0, 6.0), Variant::Vector3(7.0, 8.0, 9.0)), json!({"type": "Basis", "args": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]}));
        test(Variant::Transform2D(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), json!({"type": "Transform2D", "args": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}));
        test(Variant::Transform3D(Variant::Vector3(1.0, 2.0, 3.0), Variant::Vector3(4.0, 5.0, 6.0), Variant::Vector3(7.0, 8.0, 9.0), Variant::Vector3(10.0, 11.0, 12.0)), json!({"type": "Transform3D", "args": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}));
        test(Variant::Projection(Variant::Vector4(1.0, 2.0, 3.0, 4.0), Variant::Vector4(5.0, 6.0, 7.0, 8.0), Variant::Vector4(9.0, 10.0, 11.0, 12.0), Variant::Vector4(13.0, 14.0, 15.0, 16.0)), json!({"type": "Projection", "args": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]}));
        test(Variant::Color(1.0, 2.0, 3.0, 4.0), json!({"type": "Color", "args": [1.0, 2.0, 3.0, 4.0]}));

        // Dictionary and Array
        test(Variant::Dictionary(HashMap::from([("key".to_string(), Variant::Int(1))])), json!({"type": "Dictionary", "args": [{"key": "i:1"}]}));
        test(Variant::Array(vec![Variant::Int(1), Variant::String("abc".to_string())]), json!({"type": "Array", "args": ["i:1", "s:abc"]}));
    }
}